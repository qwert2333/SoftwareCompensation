#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

from calo.experiment_grid import EXPERIMENTS, make_run_name
from calo.seed import set_global_seed
from calo.train import train_one
from calo.evaluate import evaluate_one


def main(array_id: int):
    if array_id < 0 or array_id >= len(EXPERIMENTS):
        raise IndexError(f"array_id={array_id} out of range for {len(EXPERIMENTS)} experiments")

    exp = EXPERIMENTS[array_id]
    seed = 20250000 + array_id
    set_global_seed(seed)

    run_name = make_run_name(exp)
    out_dir = os.path.join("outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "array_id": array_id,
                "seed": seed,
                **exp,
            },
            f,
            indent=2,
        )

    train_one(exp=exp, out_dir=out_dir, seed=seed)
    evaluate_one(exp=exp, out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-id", type=int, required=True)
    args = ap.parse_args()
    main(args.array_id)
