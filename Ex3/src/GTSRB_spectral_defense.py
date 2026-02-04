############################
## GTSRB Spectral Defense ##
############################

#################
#### IMPORTS ####
#################
import sys
import os
import argparse
import csv
import torch
from torchvision.models import resnet18 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from GTSRB import GTSRB_Wrapper


sys.path.append(os.getcwd())
try:
    from BackdoorBox.core.defenses.Spectral import Spectral
except ImportError:
    # If first import fails, try adding parent directory to path and import again
    sys.path.append(os.path.join(os.getcwd(), '..'))
    print(sys.path)
    try:
        from BackdoorBox.core.defenses.Spectral import Spectral
    except ImportError as e:
        print("Failed to import BackdoorBox:", e)
        sys.exit(1)

#################
## IMPORTS END ##
#################

#################
#### HELPERS ####
#################
def eval_net(model, loader_clean, loader_poison, device, name="Model"):
    model.eval()
    # Clean Acc
    corr = 0
    tot = 0
    with torch.no_grad():
        for x, y, _ in loader_clean:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            corr += (pred == y).sum().item()
            tot += y.size(0)
    acc = 100 * corr / tot
    
    # ASR
    corr_p = 0
    tot_p = 0
    with torch.no_grad():
        for x, y, _ in loader_poison:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            corr_p += (pred == y).sum().item()
            tot_p += y.size(0)
    asr = 100 * corr_p / tot_p if tot_p > 0 else 0
    
    print(f"[{name}] Clean Acc: {acc:.2f}% | ASR: {asr:.2f}%")
    return acc, asr

def train_model(model, loader, device, criterion, lr, epochs):
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}")
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    return model

#################
## HELPERS END ##
#################


def main():
    parser = argparse.ArgumentParser(description="GTSRB Spectral Defense")
    parser.add_argument("--poison_type", type=str, default="black_1", choices=['black_1', 'green_0_5', 'green_1'], help="Type of poison: 'black_1', 'green_0_5', 'green_1'")
    parser.add_argument("--poison_rate", type=float, default=0.01, help="Poison injection rate (e.g., 0.01)")
    parser.add_argument("--target_label", type=int, default=5, help="Target label index")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_csv", type=str, default="spectral_defense_results.csv", help="Path to save results CSV")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    
    args = parser.parse_args()

    print("\n" + "#"*30)
    print("## GTSRB Spectral Defense ##")
    print("#"*30 + "\n")

    # Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    # Load Data
    print(f"> Loading data (Poison Type: {args.poison_type}, Rate: {args.poison_rate})...")
    train_set = GTSRB_Wrapper(root_dir=args.data_root, mode='train', poison_type=args.poison_type, poison_rate=args.poison_rate, transform=transform, target_label=args.target_label)
    test_clean = GTSRB_Wrapper(root_dir=args.data_root, mode='test', poison_type=args.poison_type, poison_rate=0.0, transform=transform, target_label=args.target_label)
    # Test set Poison: poison_rate=1.0 for ASR
    test_poison = GTSRB_Wrapper(root_dir=args.data_root, mode='test', poison_type=args.poison_type, poison_rate=1.0, transform=transform, target_label=args.target_label)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader_clean = DataLoader(test_clean, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader_poison = DataLoader(test_poison, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    poisoned_location = set()
    for i in range(len(train_set)):
        _, _, is_p = train_set[i]
        if int(is_p) == 1:
            poisoned_location.add(i)
    
    print(f"  Poisoned samples in train_set: {len(poisoned_location)} / {len(train_set)}")

    # Train Backdoored Model
    print("\n> Training Baseline Backdoored Model...")
    badnet = resnet18(num_classes=43).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    
    # Train
    badnet = train_model(badnet, train_loader, DEVICE, crit, args.lr, args.epochs)
    
    # Eval
    clean_acc, asr = eval_net(badnet, test_loader_clean, test_loader_poison, DEVICE, "Baseline")

    results = []
    # Store Baseline Results
    results.append({
        "poison_type": args.poison_type,
        "poison_rate": args.poison_rate,
        "target_label": args.target_label,
        "defense": "Baseline (BadNet)",
        "percentile": "N/A",
        "clean_acc": clean_acc,
        "asr": asr,
        "precision": "N/A",
        "recall": "N/A"
    })

    # Spectral Defense
    print("\n" + "#"*20)
    print("## Spectral Defense ##")
    print("#"*20)

    schedule = {
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "GPU_num": 1,
        "num_workers": args.num_workers
    }
    
    percentiles = [80, 82, 85]

    for p in percentiles:
        print(f"\n> Running Spectral Defense (Percentile={p})...")
        spectral = Spectral(
            model=badnet,
            loss=crit,
            seed=0,
            target_label=args.target_label,
            percentile=p,
            poisoned_trainset=train_set
        )
        removed_global, kept_global = spectral.filter(schedule)

        # Metrics
        T = set(i for i in range(len(train_set)) if int(train_set[i][1]) == args.target_label)
        P = poisoned_location & T
        Pred = set(removed_global.tolist())

        tp = len(Pred & P)
        fp = len(Pred - P)
        fn = len(P - Pred)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        
        print(f"  Removed: {len(Pred)} | TP: {tp} | FP: {fp} | FN: {fn}")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f}")

        # Retrain
        print("  Retraining on filtered set...")
        filtered_train = Subset(train_set, kept_global)
        filtered_loader = DataLoader(filtered_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        defended = resnet18(num_classes=43).to(DEVICE)
        defended = train_model(defended, filtered_loader, DEVICE, crit, args.lr, args.epochs)
        
        d_acc, d_asr = eval_net(defended, test_loader_clean, test_loader_poison, DEVICE, f"Defended p={p}")
        
        results.append({
            "poison_type": args.poison_type,
            "poison_rate": args.poison_rate,
            "target_label": args.target_label,
            "defense": "Spectral",
            "percentile": p,
            "clean_acc": d_acc,
            "asr": d_asr,
            "precision": precision,
            "recall": recall
        })

    # Save to CSV
    print(f"\n> Saving results to {args.output_csv}...")
    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print("Done.")

if __name__ == "__main__":
    main()
