# config.py
# Define experiments as dictionaries. run_all.py will convert them into CLI token lists
# and append a unique --output_csv automatically per run.

PYTHON = "python3"
SRC_DIR = "src"
RESULTS_DIR = "results"
ADD_TIMESTAMP_TO_OUTPUT = True

# ----------------------------
# Experiments
# ----------------------------
# Each experiment dict must include:
#   name: unique identifier for humans
#   defense: "spectral" or "ae"
#   dataset: "gtsrb" or "yf"
#   poison_type: e.g. black_1, green_0_5, green_1, beard, glasses
#   args: flat dict of CLI args (strings / numbers / lists)
#
# Notes:
# - Lists are expanded as repeated tokens (e.g. alphas -> "--alphas a b c")
# - Do NOT include output_csv here; run_all.py sets it per experiment.

EXPERIMENTS = [
    # ----------------------------
    # Autoencoder - GTSRB
    # ----------------------------
    {
        "name": "ae__gtsrb__black_1__pr0.01__t5",
        "defense": "ae",
        "dataset": "gtsrb",
        "poison_type": "black_1",
        "args": {
            "dataset": "gtsrb",
            "poison_type": "black_1",
            "poison_rate": 0.01,
            "target_label": 5,
            "batch_size": 256,
            "ae_batch": 256,
            "num_workers": 8,
            "lr": 0.01,
            "alphas": [0.65, 0.80, 1.00],
        },
    },
    {
        "name": "ae__gtsrb__green_0_5__pr0.01__t5",
        "defense": "ae",
        "dataset": "gtsrb",
        "poison_type": "green_0_5",
        "args": {
            "dataset": "gtsrb",
            "poison_type": "green_0_5",
            "poison_rate": 0.01,
            "target_label": 5,
            "batch_size": 256,
            "ae_batch": 256,
            "num_workers": 8,
            "lr": 0.01,
            "alphas": [0.65, 0.80, 1.00],
        },
    },
    {
        "name": "ae__gtsrb__green_1__pr0.01__t5",
        "defense": "ae",
        "dataset": "gtsrb",
        "poison_type": "green_1",
        "args": {
            "dataset": "gtsrb",
            "poison_type": "green_1",
            "poison_rate": 0.01,
            "target_label": 5,
            "batch_size": 256,
            "ae_batch": 256,
            "num_workers": 8,
            "lr": 0.01,
            "alphas": [0.65, 0.80, 0.9],
        },
    },

    # ----------------------------
    # Autoencoder - YaleFaces
    # ----------------------------
    {
        "name": "ae__yf__beard__pr0.05__t7__a0.5_0.6_0.7",
        "defense": "ae",
        "dataset": "yf",
        "poison_type": "beard",
        "args": {
            "dataset": "yf",
            "poison_type": "beard",
            "poison_rate": 0.05,
            "target_label": 7,
            "batch_size": 256,
            "ae_batch": 256,
            "num_workers": 8,
            "lr": 0.01,
            "alphas": [0.5, 0.6, 0.7],
        },
    },
    {
        "name": "ae__yf__glasses__pr0.05__t8__a0.4_0.6_0.8",
        "defense": "ae",
        "dataset": "yf",
        "poison_type": "glasses",
        "args": {
            "dataset": "yf",
            "poison_type": "glasses",
            "poison_rate": 0.05,
            "target_label": 8,
            "batch_size": 128,
            "ae_batch": 256,
            "num_workers": 8,
            "lr": 0.01,
            "alphas": [0.6, 0.75, 0.9]
        },
    },

    # ----------------------------
    # Spectral - GTSRB
    # ----------------------------
    {
        "name": "spectral__gtsrb__black_1__pr0.01__t5",
        "defense": "spectral",
        "dataset": "gtsrb",
        "poison_type": "black_1",
        "args": {
            "dataset": "gtsrb",
            "poison_type": "black_1",
            "poison_rate": 0.01,
            "target_label": 5,
            "batch_size": 128,
            "num_workers": 8,
            "lr": 0.01,
            "epochs": 5,
        },
    },
    {
        "name": "spectral__gtsrb__green_0_5__pr0.01__t5",
        "defense": "spectral",
        "dataset": "gtsrb",
        "poison_type": "green_0_5",
        "args": {
            "dataset": "gtsrb",
            "poison_type": "green_0_5",
            "poison_rate": 0.01,
            "target_label": 5,
            "batch_size": 256,
            "num_workers": 8,
            "lr": 0.01,
            "epochs": 5,
        },
    },
    {
        "name": "spectral__gtsrb__green_1__pr0.01__t5",
        "defense": "spectral",
        "dataset": "gtsrb",
        "poison_type": "green_1",
        "args": {
            "dataset": "gtsrb",
            "poison_type": "green_1",
            "poison_rate": 0.01,
            "target_label": 5,
            "batch_size": 128,
            "num_workers": 8,
            "lr": 0.01,
            "epochs": 5,
        },
    },

    # ----------------------------
    # Spectral - YaleFaces
    # ----------------------------
    {
        "name": "spectral__yf__beard__pr0.02__t7",
        "defense": "spectral",
        "dataset": "yf",
        "poison_type": "beard",
        "args": {
            "dataset": "yf",
            "poison_type": "beard",
            "poison_rate": 0.02,
            "target_label": 7,
            "batch_size": 128,
            "num_workers": 8,
            "lr": 0.005,
            "epochs": 10,
        },
    },
    {
        "name": "spectral__yf__glasses__pr0.02__t8",
        "defense": "spectral",
        "dataset": "yf",
        "poison_type": "glasses",
        "args": {
            "dataset": "yf",
            "poison_type": "glasses",
            "poison_rate": 0.02,
            "target_label": 8,
            "batch_size": 128,
            "num_workers": 8,
            "lr": 0.01,
            "epochs": 10,
        },
    },
]
