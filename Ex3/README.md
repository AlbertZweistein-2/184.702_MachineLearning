# Spectral Signature Defense

This module implements the **Spectral Signature Defense** for detecting poisoned data in image classification tasks.

## Overview

The defense can be evaluated by running the `spectral_defense.py` script, which trains a model on a specified dataset and analyzes hidden representations to identify potential backdoor or poisoning attacks.

## Usage

Run the script from the `src` directory:

```bash
python src/spectral_defense.py [OPTIONS]
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|------------|----------|
| `--dataset {gtrsb,yf}` | Dataset to use: **GTSRB** (German Traffic Signs) or **Yale Faces** | `gtrsb` |
| `--poison_type {black_1,green_0_5,green_1,beard,glasses}` | Type of poisoning or trigger applied | `black_1` |
| `--poison_rate POISON_RATE` | Fraction of poisoned samples (value between 0 and 1) | 0.01 |
| `--target_label TARGET_LABEL` | Target label for the attack | 5 |
| `--data_root DATA_ROOT` | Root directory containing the dataset | `data` |
| `--output_csv OUTPUT_CSV` | Path to save the output CSV results | `spectral_defense_results.csv` |
| `--epochs EPOCHS` | Number of training epochs | 15 |
| `--batch_size BATCH_SIZE` | Training batch size | 64 |
| `--lr LR` | Learning rate | 0.01 |
| `--num_workers NUM_WORKERS` | Number of data loading workers | 4 |


## Example

```bash
python src/spectral_defense.py \
  --dataset "gtrsb" \
  --poison_type "black_1" \
  -- data_root "../data"
  --poison_rate 0.1 \
  --target_label 0 \
  --epochs 20 \
  --batch_size 64
```

# Autoencoder
