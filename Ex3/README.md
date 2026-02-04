# Folder Structure

This presents the necessary folder structure for the modules to work correctly.

```
Ex3/
├── BackdoorBox/ # models
│
├── data/ # Directory for datasets
│ ├── Faces/
│ │ ├── beard/
│ │ │ ├── beard_extended/
│ │ │ └── beard_test_extended/
│ │ ├── extendedData/
│ │ │ ├── original_extended/
│ │ │ └── original_test_extended/
│ │ └── glasses/
│ │ ├── glasses_extended/
│ │ └── glasses_test_extended/
│ │
│ ├── GTSRB/
│ │ ├── original/
│ │ │ ├── Meta/
│ │ │ ├── Train/
│ │ │ ├── Test/
│ │ │ ├── Meta.csv
│ │ │ ├── Train.csv
│ │ │ └── Test.csv
│ │ │
│ │ └── poisoned/
│ │ ├── GTSRB_backdoor_black_1/
│ │ ├── GTSRB_backdoor_green_0_5/
│ │ └── GTSRB_backdoor_green_1/
│
├── src/ # All training/defense scripts
│
├── papers/ # PDFs / literature
│
├── README.md
└── .gitignore

```

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
| `--dataset {gtsrb,yf}` | Dataset to use: **GTSRB** (German Traffic Signs) or **Yale Faces** | `gtsrb` |
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
  --dataset "gtsrb" \
  --poison_type "black_1" \
  -- data_root "../data"
  --poison_rate 0.1 \
  --target_label 0 \
  --epochs 20 \
  --batch_size 64
```

# Autoencoder

This module implements the **Autoencoder Preprocessing Defense** for mitigating backdoors in poisoned datasets.

## Overview

The defense can be evaluated by running the `autoencoder_defense.py` script, which trains a model on a poisoned dataset and an autoencoder on clean data.

## Usage

Run the script from the `src` directory:

```bash
python src/autoencoder_defense.py [OPTIONS]
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|------------|----------|
| `--dataset {gtsrb,yf}` | Dataset to use: **GTSRB** (German Traffic Signs) or **Yale Faces** | `gtsrb` |
| `--poison_type {black_1,green_0_5,green_1,beard,glasses}` | Type of poisoning or trigger applied | `black_1` |
| `--poison_rate POISON_RATE` | Fraction of poisoned samples (value between 0 and 1) | 0.01 |
| `--target_label TARGET_LABEL` | Target label for the attack | 5 |
| `--data_root DATA_ROOT` | Root directory containing the dataset | `data` |
| `--output_csv OUTPUT_CSV` | Path to save the output CSV results | `spectral_defense_results.csv` |
| `--seed SEED` | Seed | 0 |
| `--epochs EPOCHS` | Number of training epochs | 15 |
| `--batch_size BATCH_SIZE` | Training batch size | 64 |
| `--lr LR` | Learning rate | 0.01 |
| `--num_workers NUM_WORKERS` | Number of data loading workers | 4 |
| `--ae_epochs AE_EPOCHS` | Number of training epochs for autoencoder | 20 |
| `--ae_lr AE_LR` | Learning rate for autoencoder | 1e-3 |
| `--ae_batch AE_BATCH_SIZE` | Training batch size for autoencoder | 64 |
| `--alphas ALPHAS` | 3 versions via alpha | [0.65, 0.80, 1.00] |

## Example

```bash
python autoencoder_defense.py --dataset "yf" --poison_type "beard" 
```
