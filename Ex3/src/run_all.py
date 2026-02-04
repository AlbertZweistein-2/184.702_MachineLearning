#!/usr/bin/env python3
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import config as cfg


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def dict_to_tokens(args: dict) -> list[str]:
    """
    Convert {"batch_size":256, "alphas":[0.4,0.6,0.8]} to:
      ["--batch_size","256","--alphas","0.4","0.6","0.8"]
    """
    tokens: list[str] = []
    for k, v in args.items():
        flag = f"--{k}"
        if isinstance(v, (list, tuple)):
            tokens.append(flag)
            tokens.extend([str(x) for x in v])
        else:
            tokens.extend([flag, str(v)])
    return tokens


def build_output_path(exp: dict, script_stem: str, results_dir: Path) -> Path:
    name = exp.get("name", "exp")
    if cfg.ADD_TIMESTAMP_TO_OUTPUT:
        name = f"{name}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    fname = sanitize(f"{script_stem}__{name}") + ".csv"
    return results_dir / fname


def run_one(script_path: Path, exp: dict, results_dir: Path):
    script_stem = script_path.stem
    out_csv = build_output_path(exp, script_stem, results_dir)

    arg_tokens = dict_to_tokens(exp["args"])

    # safety: remove any accidental output_csv
    cleaned = []
    skip_next = False
    for tok in arg_tokens:
        if skip_next:
            skip_next = False
            continue
        if tok == "--output_csv":
            skip_next = True
            continue
        cleaned.append(tok)

    cmd = [cfg.PYTHON, str(script_path), *cleaned, "--output_csv", str(out_csv)]

    print("\n" + "=" * 120)
    print("NAME   :", exp.get("name", "exp"))
    print("DEFENSE:", exp.get("defense"))
    print("DATASET:", exp.get("dataset"))
    print("POISON :", exp.get("poison_type"))
    print("CMD    :", " ".join(cmd))
    print("OUT    :", out_csv)
    print("=" * 120)

    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"FAILED ({res.returncode}): {' '.join(cmd)}")


def matches_filters(exp: dict, defense: str | None, dataset: str | None, poison: str | None) -> bool:
    if defense and exp.get("defense") != defense:
        return False
    if dataset and exp.get("dataset") != dataset:
        return False
    if poison and exp.get("poison_type") != poison:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run experiment batches with optional filters.")
    parser.add_argument("--defense", choices=["spectral", "ae"], default=None,
                        help="Run only one defense type")
    parser.add_argument("--dataset", choices=["gtsrb", "yf"], default=None,
                        help="Run only one dataset")
    parser.add_argument("--poison", default=None,
                        help="Run only one poison_type (e.g., black_1, green_0_5, green_1, beard, glasses)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent  # Ex3/ (parent of src/)

    src_dir = repo_root / cfg.SRC_DIR
    results_dir = repo_root / cfg.RESULTS_DIR
    ensure_dir(results_dir)

    ae_script = src_dir / "autoencoder_defense.py"
    sp_script = src_dir / "spectral_defense.py"

    # filter experiments
    selected = [
        exp for exp in cfg.EXPERIMENTS
        if matches_filters(exp, args.defense, args.dataset, args.poison)
    ]

    if not selected:
        print("No experiments matched your filters.")
        return

    print(f"\nSelected experiments: {len(selected)}")
    if args.defense: print(" - defense:", args.defense)
    if args.dataset: print(" - dataset:", args.dataset)
    if args.poison:  print(" - poison :", args.poison)

    for exp in selected:
        defense = exp["defense"]
        if defense == "ae":
            run_one(ae_script, exp, results_dir)
        elif defense == "spectral":
            run_one(sp_script, exp, results_dir)
        else:
            raise ValueError(f"Unknown defense in experiment: {defense}")

    print("\nDone. All outputs are in:", results_dir)


if __name__ == "__main__":
    main()
