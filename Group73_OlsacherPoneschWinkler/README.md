# 1 Folder Structure

This repository expects the following folder structure for all modules to work correctly:

```
Ex3/
├── BackdoorBox/ # BackdoorBox reference implementation [1]
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
│ │   ├── glasses_extended/
│ │   └── glasses_test_extended/
│ │
│ └── GTSRB/
│   ├── original/
│   │ ├── Meta/
│   │ ├── Train/
│   │ ├── Test/
│   │ ├── Meta.csv
│   │ ├── Train.csv
│   │ └── Test.csv
│   │
│   └── poisoned/
│     ├── GTSRB_backdoor_black_1/
│     │ ├── Test_backdoor_black_1_percent/
│     │ ├── Training_backdoor_black_1_percent/
│     ├── GTSRB_backdoor_green_0_5/
│     │ ├── Test_backdoor_green_0_5_percent/
│     │ └── Training_backdoor_green_0_5_percent/
│     └── GTSRB_backdoor_green_1/
│       ├── Test_backdoor_green_1_percent/
│       └── Training_backdoor_green_1_percent/
│ 
├── results/ # Directory for results
├── Makefile
├── src/ # All training/defense scripts
│ ├── requirements.txt
│ ├── config.py
│ ├── run_all.py
│ └── ... # other source files
│
├── papers/ # PDFs / literature
│
├── README.md
└── .gitignore
```

---

## 2 Configurations

The configurations for all experiments are defined in `src/config.py`. Each experiment is represented as a dictionary containing the following keys:

* `name`: A unique identifier for the experiment.
* `defense`: The defense method to be used (`spectral` or `ae`).
* `dataset`: The dataset to be used (`gtsrb` or `yf`).
* `poison_type`: The type of poisoning/trigger to be applied (e.g., `black_1`, `green_0_5`, `green_1`, `beard`, `glasses`).
* `args`: A dictionary of additional command-line arguments required for the experiment.

---

## 3 Run All

`run_all.py` evaluates two defenses for detecting poisoned samples in image classification datasets:

* **Spectral Signature Defense**
* **Autoencoder Preprocessing Defense**

### 3.1 What the script does

When you run `run_all.py`, it:

1. Loads the selected dataset (GTSRB or Yale Faces)
2. Applies the selected poisoning/trigger variant (depending on the dataset)
3. Trains a model (or loads a model if implemented that way in your code)
4. Runs the selected defense method
5. Prints an overview of results for the chosen configuration

### 3.2 Download datasets

* **Original German Traffic Sign (GTSRB)**: [https://www.kaggle.com/datasets/meowmeowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
* **Poisoned German Traffic Sign**: [https://zenodo.org/record/3716766](https://zenodo.org/record/3716766)
* **Yale Face**: [https://zenodo.org/records/3774167](https://zenodo.org/records/3774167)

Make sure you have the required datasets in the `data/` folder as shown in the folder structure above.

### 3.3 Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or use the Makefile:

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

### 3.4 Usage

Run the script from the `Ex3` directory by either using the Makefile:

```bash
make run
```

You can also run it directly:

```bash
python src/run_all.py [OPTIONS]
```

### 3.5 Command line arguments

| Argument    | Choices                                               | Description                                                        | Default    |
| ----------- | ----------------------------------------------------- | ------------------------------------------------------------------ | ---------- |
| `--defense` | `spectral`, `ae`                                      | Which defense method to use                                        | `spectral` |
| `--dataset` | `gtsrb`, `yf`                                         | Dataset to use: **GTSRB** (German Traffic Signs) or **Yale Faces** | `gtsrb`    |
| `--poison`  | `black_1`, `green_0_5`, `green_1`, `beard`, `glasses` | Poisoning / trigger type (valid options depend on dataset)         | `black_1`  |

#### 3.5.1 Notes on `--poison`

* For `--dataset gtsrb`, typical poison options are: `black_1`, `green_0_5`, `green_1`
* For `--dataset yf`, typical poison options are: `beard`, `glasses`

If you pass a poison type that does not exist for the selected dataset, the script should error out (recommended behavior).

### 3.6 Examples

Spectral defense on GTSRB with black trigger:

```bash
python src/run_all.py --defense spectral --dataset gtsrb --poison black_1
```

Autoencoder defense on GTSRB with green trigger:

```bash
python src/run_all.py --defense ae --dataset gtsrb --poison green_1
```

Spectral defense on Yale Faces with glasses:

```bash
python src/run_all.py --defense spectral --dataset yf --poison glasses
```

---

## 4 Additional Information

All implementations can be found in the `src/` folder.

In this project, the open-source GitHub repository [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox) was used as a reference for implementing the defenses. The original code has been adapted to fit the requirements of this exercise and more modern versions of libraries.

## References

[1] Yiming Li et al. “BackdoorBox: A Python Toolbox for Backdoor Learning”. In: ICLR Workshop. 2023. url: https://github.com/THUYimingLi/BackdoorBox.