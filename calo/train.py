import glob
import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import MultiFileCaloIterable
from .features import aux_dim
from .geo import build_geo
from .losses import compute_loss
from .model import TimeAwareFEMNet
from .voxel import channels_for

TRAIN_SHARDS = "/gpfs/workdir/xiax/Data_V0/Train_Samples/fixed_direction/pi+/shards_fixed_y/*.h5"


def run_epoch(model, loader, optim, device, *, training=True, scaler=None, weight_cfg=None, denom_min=1.0, loss_name="smooth_l1_relative", sigma_cfg=None):
    model.train() if training else model.eval()
    total, n = 0.0, 0
    data_time = 0.0
    compute_time = 0.0
    wait_t0 = time.perf_counter()

    with torch.set_grad_enabled(training):
        for batch in loader:
            data_time += time.perf_counter() - wait_t0
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            if training:
                optim.zero_grad(set_to_none=True)

            compute_t0 = time.perf_counter()
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=True):
                    pred = model(batch["ecal"], batch["hcal"], batch["aux"])
                    loss = compute_loss(
                        pred,
                        batch["energy_true"].squeeze(1),
                        weight_cfg,
                        denom_min=denom_min,
                        loss_name=loss_name,
                        sigma_cfg=sigma_cfg,
                    )
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
            else:
                pred = model(batch["ecal"], batch["hcal"], batch["aux"])
                loss = compute_loss(
                    pred,
                    batch["energy_true"].squeeze(1),
                    weight_cfg,
                    denom_min=denom_min,
                    loss_name=loss_name,
                    sigma_cfg=sigma_cfg,
                )
                if training:
                    loss.backward()
                    optim.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time += time.perf_counter() - compute_t0

            bs = batch["ecal"].size(0)
            total += float(loss.item()) * bs
            n += bs
            wait_t0 = time.perf_counter()

    return total / max(n, 1), data_time, compute_time


def run_epoch_mixed(
    model,
    loaders,
    probs,
    max_steps,
    optim,
    device,
    *,
    training=True,
    scaler=None,
    weight_cfg=None,
    denom_min=1.0,
    loss_name="smooth_l1_relative",
    sigma_cfg=None,
):
    model.train() if training else model.eval()
    total, n = 0.0, 0
    data_time = 0.0
    compute_time = 0.0
    rng = np.random.default_rng(20260316)
    names = list(loaders.keys())
    p = np.asarray([probs[name] for name in names], dtype=np.float64)
    p = p / p.sum()
    iters = {name: iter(loader) for name, loader in loaders.items()}
    wait_t0 = time.perf_counter()

    with torch.set_grad_enabled(training):
        for _ in range(max_steps):
            pick = names[int(rng.choice(len(names), p=p))]
            try:
                batch = next(iters[pick])
            except StopIteration:
                iters[pick] = iter(loaders[pick])
                batch = next(iters[pick])
            data_time += time.perf_counter() - wait_t0
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            if training:
                optim.zero_grad(set_to_none=True)
            compute_t0 = time.perf_counter()
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=True):
                    pred = model(batch["ecal"], batch["hcal"], batch["aux"])
                    loss = compute_loss(
                        pred,
                        batch["energy_true"].squeeze(1),
                        weight_cfg,
                        denom_min=denom_min,
                        loss_name=loss_name,
                        sigma_cfg=sigma_cfg,
                    )
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
            else:
                pred = model(batch["ecal"], batch["hcal"], batch["aux"])
                loss = compute_loss(
                    pred,
                    batch["energy_true"].squeeze(1),
                    weight_cfg,
                    denom_min=denom_min,
                    loss_name=loss_name,
                    sigma_cfg=sigma_cfg,
                )
                if training:
                    loss.backward()
                    optim.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time += time.perf_counter() - compute_t0
            bs = batch["ecal"].size(0)
            total += float(loss.item()) * bs
            n += bs
            wait_t0 = time.perf_counter()

    return total / max(n, 1), data_time, compute_time


def save_checkpoint(path, model, optim, scaler, epoch, best_val, patience_counter, log, exp, seed):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
            "patience_counter": patience_counter,
            "log": log,
            "exp": exp,
            "seed": seed,
        },
        path,
    )


def load_checkpoint(path, model, optim, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def plot_loss_curves(log, out_dir):
    if not log["train"]:
        return
    epochs = range(1, len(log["train"]) + 1)
    plt.figure(figsize=(6, 4.5))
    plt.plot(epochs, log["train"], label="train")
    plt.plot(epochs, log["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Validation objective")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.pdf"), dpi=150)
    plt.close()


def resolve_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No training shards found: {pattern}")
    return files


def split_files(files, train_frac=0.7):
    cut = int(len(files) * train_frac)
    return files[:cut], files[cut:]


def count_events(files):
    import h5py
    total = 0
    for path in files:
        with h5py.File(path, 'r') as f:
            total += len(f['trueParticleEnergy'])
    return total


def make_dataset(files, geo, time_mode, batch_size, split, aux_mode, include_hitcount, shuffle_files, shuffle_events, time_resolution_ns, time_boundary_mode, time_source, y_mode, **feature_cfg):
    return MultiFileCaloIterable(
        files,
        geo=geo,
        time_mode=time_mode,
        batch_size=batch_size,
        split=split,
        train_frac=0.7,
        shuffle_files=shuffle_files,
        shuffle_events=shuffle_events,
        aux_mode=aux_mode,
        include_hitcount=include_hitcount,
        time_resolution_ns=time_resolution_ns,
        time_boundary_mode=time_boundary_mode,
        time_source=time_source,
        y_mode=y_mode,
        **feature_cfg,
    )


def make_loader(dataset, device, num_workers):
    kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs['prefetch_factor'] = 2
    return DataLoader(dataset, **kwargs)


def train_one(exp: dict, out_dir: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    geo = build_geo(**exp["geo"])
    time_mode = exp["time_mode"]
    aux_mode = exp.get("aux_mode", "energy")
    include_hitcount = bool(exp.get("include_hitcount", False))
    strategy = exp.get("train_strategy", "single_dataset")
    weight_cfg = exp.get("loss_weight")
    denom_min = float(exp.get("denom_min", 1.0))
    loss_name = str(exp.get("loss_name", "smooth_l1_relative"))
    sigma_cfg = exp.get("loss_sigma")
    time_resolution_ns = float(exp.get("time_resolution_ns", 0.0))
    time_boundary_mode = str(exp.get("threshold_mode", "fixed_train_eventwise_median"))
    feature_cfg = {k: exp[k] for k in ("include_energy_channels","include_propagation","threshold_mode","oracle_features","shuffle_times")}
    time_source = str(exp.get("time_source", "corrected"))
    y_mode = str(exp.get("y_mode", "position_bin"))

    in_ch = channels_for(time_mode, include_hitcount, bool(exp.get("include_energy_channels", True)), bool(exp.get("include_propagation", False)))
    use_longitudinal_profile = bool(exp.get("use_longitudinal_profile", False))
    model = TimeAwareFEMNet(
        in_channels=in_ch,
        aux_dim=aux_dim(aux_mode, bool(exp.get("oracle_features", False))),
        use_longitudinal_profile=use_longitudinal_profile,
    ).to(device)

    lr = float(exp.get("lr", 1e-3))
    batch_size = 128
    num_workers = 4
    epochs = int(exp.get("epochs", 25))

    train_files_all = resolve_files(TRAIN_SHARDS)
    train_files, val_files = split_files(train_files_all)

    val_ds = make_dataset(
        train_files_all, geo, time_mode, batch_size, 'val', aux_mode, include_hitcount, False, False,
        time_resolution_ns, time_boundary_mode, time_source, y_mode, **feature_cfg
    )
    val_loader = make_loader(val_ds, device, max(2, num_workers // 4))

    train_ds = make_dataset(
        train_files_all, geo, time_mode, batch_size, 'train', aux_mode, include_hitcount, True, True,
        time_resolution_ns, time_boundary_mode, time_source, y_mode, **feature_cfg
    )
    train_loader = make_loader(train_ds, device, num_workers)

    print(f"Training shards: {len(train_files)} train files, {len(val_files)} validation files")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(ckpt_dir, 'last.pth')
    best_ckpt = os.path.join(ckpt_dir, 'best.pth')

    best = 1e30
    patience, pc = 5, 0
    log = {'train': [], 'val': []}
    start_epoch = 1

    if os.path.exists(last_ckpt):
        ckpt = load_checkpoint(last_ckpt, model, optim, scaler, device)
        start_epoch = int(ckpt['epoch']) + 1
        best = float(ckpt.get('best_val', best))
        pc = int(ckpt.get('patience_counter', 0))
        log = ckpt.get('log', log)
        print(f"Resuming from {last_ckpt} at epoch {start_epoch}")

    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()

        if strategy != 'single_dataset':
            raise ValueError(f'Unknown strategy for this project: {strategy}')
        tr, tr_data_s, tr_compute_s = run_epoch(
            model,
            train_loader,
            optim,
            device,
            training=True,
            scaler=scaler,
            weight_cfg=weight_cfg,
            denom_min=denom_min,
            loss_name=loss_name,
            sigma_cfg=sigma_cfg,
        )

        va, va_data_s, va_compute_s = run_epoch(
            model,
            val_loader,
            optim,
            device,
            training=False,
            scaler=scaler,
            denom_min=denom_min,
            loss_name=loss_name,
            sigma_cfg=sigma_cfg,
        )
        dt = (time.time() - t0) / 60.0

        print(
            f"Dataset: {exp['geo']} + {exp['time_mode']} ({exp.get('time_realism', 'perfect_time')}) "
            f"aux={aux_mode} hitcount={include_hitcount} longprof={use_longitudinal_profile} strategy={strategy} "
            f"y_mode={y_mode} time_source={time_source} loss={loss_name} denom_min={denom_min} "
            f"sigma={sigma_cfg} time_resolution_ns={time_resolution_ns} time_boundary_mode={time_boundary_mode} lr={lr}"
        )
        print(f"[{ep:02d}/{epochs}] {dt:.1f} min | train={tr:.4f} val={va:.4f}")
        print(
            f"           timing: train data {tr_data_s/60.0:.1f}m compute {tr_compute_s/60.0:.1f}m | "
            f"val data {va_data_s/60.0:.1f}m compute {va_compute_s/60.0:.1f}m"
        )

        log['train'].append(tr)
        log['val'].append(va)
        plot_loss_curves(log, out_dir)

        if va < best:
            best = va
            pc = 0
            save_checkpoint(best_ckpt, model, optim, scaler, ep, best, pc, log, exp, seed)
            print('  -> saved checkpoints/best.pth')
        else:
            pc += 1

        save_checkpoint(last_ckpt, model, optim, scaler, ep, best, pc, log, exp, seed)
        save_checkpoint(os.path.join(ckpt_dir, f"epoch_{ep:03d}.pth"), model, optim, scaler, ep, best, pc, log, exp, seed)

        if bool(exp.get("early_stopping", False)) and pc >= patience:
            print('  -> early stop')
            break

    with open(os.path.join(out_dir, 'loss_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
