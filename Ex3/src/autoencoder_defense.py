#!/usr/bin/env python3
"""
#################
## AE Defense  ##
#################

Pipeline (single run, like spectral):
1) Train BadNet on POISONED train set
2) Train AE on CLEAN-ONLY train set using BackdoorBox AutoEncoderDefense
3) Evaluate:
   - Baseline Clean/ASR
   - 3 variants of the SAME defense by alpha-mixing:
       x_mix = (1 - alpha) * x + alpha * AE(x)

Outputs:
- prints baseline + defended metrics
- OVERWRITES (replaces) the CSV each run

No checkpoint loading. No complex options.
"""

import os
import sys
import csv
import time
import random
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

from GTSRB import GTSRB_Wrapper
from YaleFaces import YaleFaces_Wrapper


# BackdoorBox import (robust)
sys.path.append(os.getcwd())
try:
    from BackdoorBox.core.defenses.AutoEncoderDefense import AutoEncoderDefense
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), ".."))
    try:
        from BackdoorBox.core.defenses.AutoEncoderDefense import AutoEncoderDefense
    except ImportError as e:
        print("Failed to import BackdoorBox:", e)
        sys.exit(1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform(dataset, for_ae: bool = False):
    if dataset == "gtsrb":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:  # YaleFaces
        # classifier transform (unchanged)
        if not for_ae:
            return transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            ])
        # AE transform: add blur BEFORE ToTensor/Normalize
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])



def get_badnet(num_classes: int = 43) -> nn.Module:
    # Match Spectral baseline (default ResNet18 stem)
    return resnet18(num_classes=num_classes)


class AE_BCE(nn.Module):
    """AutoEncoder compatible with BackdoorBox's BCELoss training."""
    def __init__(self):
        super(AE_BCE, self).__init__()
        # Encoder: 128x128 -> 26x26
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 7), nn.ReLU(),
        )
        # Decoder: 26x26 -> 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class TightAEv3_Drop05(nn.Module):
    """Your chosen AE architecture."""
    def __init__(self, p: float = 0.05):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(p),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(p),
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(p),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(24, 24, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 3, 3, stride=2, padding=1, output_padding=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    c, t = 0, 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        c += (pred == y).sum().item()
        t += y.size(0)
    return 100.0 * c / max(1, t)


def train_badnet(model, train_loader, device, epochs, lr):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # match spectral (no wd)
    
    print("\n=== Training BadNet (poisoned train) ===")
    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch}/{epochs}")
        model.train()
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            
    return model


@torch.no_grad()
def eval_with_alpha_mix(clf, defense, loader, device, alpha: float, dataset: str) -> float:
    clf.eval()
    defense.autoencoder.eval()
    c, t = 0, 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_rec = defense.preprocess(x)
        x_mix = (1.0 - alpha) * x + alpha * x_rec
        x_mix = torch.clamp(x_mix, -1, 1)
        pred = clf(x_mix).argmax(1)
        c += (pred == y).sum().item()
        t += y.size(0)
    return 100.0 * c / max(1, t)


def write_csv_overwrite(output_csv: str, rows: List[Dict]):
    """
    Overwrite (replace) the CSV on every run.
    """
    fieldnames = [
        "timestamp",
        "poison_type",
        "poison_rate",
        "target_label",
        "defense",
        "alpha",
        "clean_acc",
        "asr",
    ]
    extra_keys = sorted({k for r in rows for k in r.keys()} - set(fieldnames))
    fieldnames = fieldnames + extra_keys

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="GTSRB AE Defense")

    # like spectral
    parser.add_argument("--dataset", type=str, default="gtsrb", choices=['gtsrb', 'yf'], help=["Dataset: 'gtsrb', 'yf'"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--poison_type", type=str, default="black_1", choices=["black_1", "green_0_5", "green_1", "beard", "glasses"])
    parser.add_argument("--poison_rate", type=float, default=0.01)
    parser.add_argument("--target_label", type=int, default=5)

    # baseline training knobs
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)

    # AE training knobs
    parser.add_argument("--ae_epochs", type=int, default=20)
    parser.add_argument("--ae_lr", type=float, default=1e-3)
    parser.add_argument("--ae_batch", type=int, default=64)

    # exactly 3 “versions” via alpha
    parser.add_argument("--alphas", type=float, nargs=3, default=[0.65, 0.80, 1.00])

    parser.add_argument("--output_csv", type=str, default="ae_defense_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = get_device()

    print("\n" + "#" * 20)
    print("## GTSRB AE Defense ##")
    print("#" * 20 + "\n")
    print(f"> Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.dataset == 'yf':
        transform_cls = get_transform(args.dataset, for_ae=False)
        transform_ae  = get_transform(args.dataset, for_ae=True)
    else:
        transform = get_transform(args.dataset)

    if args.dataset == 'gtsrb':
        if args.poison_type not in ['black_1', 'green_0_5', 'green_1']:
            raise Exception('Invalid poison type: ' + args.poison_type)
        
        print(f"> Loading data (Poison Type: {args.poison_type}, Rate: {args.poison_rate}), target={args.target_label})...")
        # AE is trained on CLEAN ONLY
        train_clean = GTSRB_Wrapper(
            root_dir=args.data_root, mode='train', 
            poison_type=args.poison_type, poison_rate=0.0, 
            transform=transform, target_label=args.target_label)
        train_poison = GTSRB_Wrapper(
            root_dir=args.data_root, mode="train",
            poison_type=args.poison_type, poison_rate=args.poison_rate,
            transform=transform, target_label=args.target_label
        )
        test_clean = GTSRB_Wrapper(
            root_dir=args.data_root, mode='test', 
            poison_type=args.poison_type, poison_rate=0.0, 
            transform=transform, target_label=args.target_label)
        test_poison = GTSRB_Wrapper(
            root_dir=args.data_root, mode='test', 
            poison_type=args.poison_type, poison_rate=1.0, 
            transform=transform, target_label=args.target_label)
        
        num_classes = 43
        
    else:
        if args.poison_type not in ['beard', 'glasses']:
            raise Exception('Invalid poison type: ' + args.poison_type)

        print(f"> Loading data (Poison Type: {args.poison_type}, Rate: {args.poison_rate}), target={args.target_label})...")
        # AE is trained on CLEAN ONLY
        train_clean = YaleFaces_Wrapper(
            root_dir=args.data_root, mode='train', 
            poison_type=args.poison_type, poison_rate=0.0, 
            transform=transform_ae, target_label=args.target_label)
        train_poison = YaleFaces_Wrapper(
            root_dir=args.data_root, mode='train', 
            poison_type=args.poison_type, poison_rate=args.poison_rate, 
            transform=transform_cls, target_label=args.target_label)
        test_clean = YaleFaces_Wrapper(
            root_dir=args.data_root, mode='test', 
            poison_type=args.poison_type, poison_rate=0.0, 
            transform=transform_cls, target_label=args.target_label)
        test_poison = YaleFaces_Wrapper(
            root_dir=args.data_root, mode='test', 
            poison_type=args.poison_type, poison_rate=1.0, 
            transform=transform_cls, target_label=args.target_label)
        
        num_classes = 15

    train_poison_loader = DataLoader(
        train_poison, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_clean_loader = DataLoader(
        test_clean, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_poison_loader = DataLoader(
        test_poison, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 1) baseline
    badnet = get_badnet(num_classes) # TODO CNN
    badnet = train_badnet(
        badnet, train_poison_loader,
        device=device, epochs=args.epochs, lr=args.lr
    )

    baseline_clean = eval_acc(badnet, test_clean_loader, device)
    baseline_asr = eval_acc(badnet, test_poison_loader, device)
    print(f"\n[Baseline] Clean={baseline_clean:.2f}% | ASR={baseline_asr:.2f}%")

    # 2) train AE (BackdoorBox)
    ae = TightAEv3_Drop05() if args.dataset == 'gtsrb' else AE_BCE()
    defense = AutoEncoderDefense(autoencoder=ae, seed=args.seed)

    schedule = {
        "device": "GPU" if device.type == "cuda" else "CPU",
        "GPU_num": 1,
        "batch_size": args.ae_batch,
        "num_workers": args.num_workers,
        "lr": args.ae_lr,
        "epochs": args.ae_epochs,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0, # TODO 1e-5
        "amsgrad": False,
        "log_iteration_interval": 200,
        "test_epoch_interval": 5, # TODO 10
        "save_epoch_interval": 999999,  # don't spam checkpoints TODO 10
        "save_dir": "ae_logs",
        "experiment_name": "gtsrb_ae_simple_retrain",
        "schedule": [10, 15],  # mild LR drops (for AE only)
        "gamma": 0.1,
    }

    print("\n=== Training AE (clean-only) ===")
    defense.train_autoencoder(train_clean, test_clean, schedule)

    # 3) eval 3 alphas + write CSV ONCE (overwrite)
    ts = time.strftime("%Y-%m-%d_%H:%M:%S")
    rows: List[Dict] = []

    def add_row(defense_name: str, alpha: float, clean: float, asr: float):
        rows.append({
            "timestamp": ts,
            "poison_type": args.poison_type,
            "poison_rate": args.poison_rate,
            "target_label": args.target_label,
            "defense": defense_name,
            "alpha": alpha,
            "clean_acc": clean,
            "asr": asr,
        })

    add_row("Baseline", float("nan"), float(baseline_clean), float(baseline_asr))

    print("\nalpha | Clean% | ASR%")
    for alpha in args.alphas:
        c = eval_with_alpha_mix(badnet, defense, test_clean_loader, device, alpha=float(alpha), dataset=args.dataset)
        a = eval_with_alpha_mix(badnet, defense, test_poison_loader, device, alpha=float(alpha), dataset=args.dataset)
        print(f"{alpha:>5.2f} | {c:>6.2f} | {a:>6.2f}")
        add_row("AE_Mix", float(alpha), float(c), float(a))

    write_csv_overwrite(args.output_csv, rows)
    print(f"\n> Saved results to: {args.output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
