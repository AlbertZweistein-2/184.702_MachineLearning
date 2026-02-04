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

# `run_all.py`

This module implements both **Spectral Signature Defense** and **Autoencoder Preprocessing Defense** for detecting poisoned data in image classification tasks.

## Overview

The defenses can be evaluated by running the `run_all.py` script, which trains a model or models on a specified dataset and  gives an overview of the different defense methods and datasets

## Download

Original German Traffic Sign: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Poisoned German Traffic Sign: https://zenodo.org/record/3716766


Yale Face: https://zenodo.org/records/3774167

## Usage



Run the script from the `Ex3` directory:

```bash
python src/run_all.py [OPTIONS]
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|------------|----------|
| `--defense {spectral,ae}` | Which defense methode to use | `spectral` |
| `--dataset {gtsrb,yf}` | Dataset to use: **GTSRB** (German Traffic Signs) or **Yale Faces** | `gtsrb` |
| `--poison {black_1,green_0_5,green_1,beard,glasses}` | Type of poisoning or trigger applied | `black_1` |



## Example

```bash
python3 ./src/run_all.py 
--defense spectral 
--datase gtsrb 
--poison black_1
```