"""Training loop for the two controlled dual-readout ablations."""

import json
import os
import time
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .dr_dataset import DualReadoutIterable
from .dr_model import DualReadoutHCALNet
from .device import select_device
from .dual_readout import AUX_NAMES, DualReadoutCalibration, DualReadoutGeometry
from .losses import compute_loss


def make_loader(files, exp, split, geo, calibration, device, max_events_per_file=-1):
    dataset = DualReadoutIterable(
        files, exp["mode"], split, geo, calibration, exp["split_seed"],
        shuffle=(split == "train"),
        max_events_per_file=max_events_per_file,
        cell_signal_scale_gev=exp["cell_signal_scale_gev"],
    )
    workers = int(exp.get("num_workers", 2))
    kwargs = dict(
        dataset=dataset,
        batch_size=int(exp.get("batch_size", 8)),
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def run_epoch(model, loader, optimizer, scaler, device, exp, training):
    model.train(training)
    total_loss = 0.0
    total_events = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            voxel = batch["voxel"].to(device, non_blocking=True)
            aux = batch["aux"].to(device, non_blocking=True)
            target = batch["energy_true"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", enabled=True)
                if device.type == "cuda" else nullcontext()
            )
            with amp_context:
                prediction = model(voxel, aux)
                loss = compute_loss(
                    prediction, target,
                    denom_min=float(exp.get("denom_min", 0.7)),
                    loss_name=exp.get("loss_name", "l1_relative"),
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            n = target.numel()
            total_loss += float(loss.item()) * n
            total_events += n
    if total_events == 0:
        raise RuntimeError(f"No events were read for {'train' if training else 'validation'}")
    return total_loss / total_events, total_events


def _plot(log, out_dir):
    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot(log["train"], label="train")
    axis.plot(log["val"], label="validation")
    axis.set(xlabel="Epoch", ylabel="Relative L1")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.pdf"))
    plt.close(fig)


def train_one(exp, files, out_dir, seed, max_events_per_file=-1):
    device = select_device(exp.get("device", "auto"))
    print(f"Compute device: {device}", flush=True)
    geo = DualReadoutGeometry(**exp["geometry"])
    calibration = DualReadoutCalibration(**exp["calibration"])
    model = DualReadoutHCALNet(aux_dim=len(AUX_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(exp.get("lr", 1.0e-3)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    train_loader = make_loader(files, exp, "train", geo, calibration, device, max_events_per_file)
    val_loader = make_loader(files, exp, "val", geo, calibration, device, max_events_per_file)

    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best.pth")
    last_path = os.path.join(checkpoint_dir, "last.pth")
    best = float("inf")
    start_epoch = 1
    log = {"train": [], "val": []}
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val"])
        log = checkpoint["log"]

    total_epochs = int(exp.get("epochs", 40))
    for epoch in range(start_epoch, total_epochs + 1):
        started = time.time()
        train_loss, n_train = run_epoch(model, train_loader, optimizer, scaler, device, exp, True)
        val_loss, n_val = run_epoch(model, val_loader, optimizer, scaler, device, exp, False)
        log["train"].append(train_loss)
        log["val"].append(val_loss)
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch, "best_val": min(best, val_loss), "log": log,
            "exp": exp, "seed": seed, "aux_names": AUX_NAMES,
        }
        improved = val_loss < best
        if improved:
            best = val_loss
            state["best_val"] = best
            torch.save(state, best_path)
        torch.save(state, last_path)
        epoch_checkpoint = f"epoch_{epoch:03d}.pth"
        torch.save(state, os.path.join(checkpoint_dir, epoch_checkpoint))
        _plot(log, out_dir)
        elapsed_minutes = (time.time() - started) / 60.0
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:03d}/{total_epochs:03d}] "
            f"train_loss={train_loss:.6f} n_train={n_train} "
            f"val_loss={val_loss:.6f} n_val={n_val} "
            f"best_val={best:.6f}{'*' if improved else ''} "
            f"lr={learning_rate:.3e} device={device.type} "
            f"time={elapsed_minutes:.1f}min checkpoint={epoch_checkpoint}",
            flush=True,
        )
    with open(os.path.join(out_dir, "loss_log.json"), "w") as handle:
        json.dump(log, handle, indent=2)
    return model
