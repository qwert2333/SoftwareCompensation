#!/usr/bin/env python3
"""Build resolution tables and fits from an existing dual-readout run."""

import argparse
from pathlib import Path

from calo.dr_performance import run_performance_validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs_dual_readout/Sapphire_5-5-5mm"),
    )
    parser.add_argument("--calice-local-sc-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    points, fits = run_performance_validation(
        args.run_dir, args.calice_local_sc_dir, args.output_dir
    )
    print(f"Wrote {len(points)} resolution points and {len(fits)} fits.")
    for row in fits.itertuples(index=False):
        print(
            f"[{row.method}] a={row.stochastic_term_percent_sqrtGeV:.3f}% sqrt(GeV), "
            f"b={row.constant_term_percent:.3f}%, "
            f"chi2/ndf={row.chi2:.2f}/{row.ndf}, quality={row.fit_quality}"
        )


if __name__ == "__main__":
    main()
