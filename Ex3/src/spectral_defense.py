############################
## Spectral Defense ##
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
from tqdm import tqdm

from GTSRB import GTSRB_Wrapper
from YaleFaces import YaleFaces_Wrapper
from torchvision.models import resnet18, ResNet18_Weights


sys.path.append(os.getcwd())

try:
    from BackdoorBox.core.defenses.Spectral import Spectral
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    try:
        from BackdoorBox.core.defenses.Spectral import Spectral
    except ImportError as e:
        print("Failed to import BackdoorBox:", e)
        sys.exit(1)

#################
#### HELPERS ####
#################
def eval_net(model, loader_clean, loader_poison, device, name="Model"):
    model.eval()
    
    clean_correct = 0
    clean_total = 0
    poison_correct = 0
    poison_total = 0
    
    with torch.no_grad():
        for x, y, _ in tqdm(loader_clean, desc="Evaluation Clean Accuracy", leave=False):
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            clean_correct += (pred == y).sum().item()
            clean_total += y.size(0)
        
        for x, y, _ in tqdm(loader_poison, desc="Evaluating Poisoned", leave=False):
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            poison_correct += (pred == y).sum().item()
            poison_total += y.size(0)
            
    clean_acc = 100 * clean_correct / clean_total  
    poison_acc = 100 * poison_correct / poison_total if poison_total > 0 else 0
    
    print(f"[{name}] Clean Acc: {clean_acc:.2f}% | ASR: {poison_acc:.2f}%")
    return clean_acc, poison_acc

def train_model(model, loader, device, criterion, lr, epochs):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0
    
        for x, y, _ in tqdm(loader, desc="Training", leave=False):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total  
        print(f"    - Loss: {epoch_loss:.2f}% | Accuracy: {epoch_acc:.2f}%")
    
    return model

####################
## IMPLEMENTATION ##
####################

def main():
    parser = argparse.ArgumentParser(description="Spectral Defense")
    parser.add_argument("--dataset", type=str, default="gtsrb", choices=['gtsrb', 'yf'], help=["Dataset: 'gtsrb', 'yf'"])
    parser.add_argument("--poison_type", type=str, default="black_1", 
                        choices=['black_1', 'green_0_5', 'green_1', 'beard', 'glasses'], 
                        help="Type of poison: GTRSB: 'black_1', 'green_0_5', 'green_1' (for gtrsb); YF: 'beard', 'glasses'")
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
    print(f"## Spectral Defense: {args.dataset} ##")
    print("#"*30 + "\n")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.dataset == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

    if args.dataset == 'gtsrb':
        if args.poison_type not in ['black_1', 'green_0_5', 'green_1']:
            raise Exception('Invalid poison type: ' + args.poison_type)
        
        print(f"> Loading data (Poison Type: {args.poison_type}, Rate: {args.poison_rate})...")
        train_set = GTSRB_Wrapper(root_dir=args.data_root, mode='train', poison_type=args.poison_type, poison_rate=args.poison_rate, transform=transform, target_label=args.target_label)
        test_clean = GTSRB_Wrapper(root_dir=args.data_root, mode='test', poison_type=args.poison_type, poison_rate=0.0, transform=transform, target_label=args.target_label)
        
        test_poison = GTSRB_Wrapper(root_dir=args.data_root, mode='test', poison_type=args.poison_type, poison_rate=1.0, transform=transform, target_label=args.target_label)
    else:
        if args.poison_type not in ['beard', 'glasses']:
            raise Exception('Invalid poison type: ' + args.poison_type)

        print(f"> Loading data (Poison Type: {args.poison_type}, Rate: {args.poison_rate})...")
        train_set = YaleFaces_Wrapper(root_dir=args.data_root, mode='train', poison_type=args.poison_type, poison_rate=args.poison_rate, transform=transform, target_label=args.target_label)
        test_clean = YaleFaces_Wrapper(root_dir=args.data_root, mode='test', poison_type=args.poison_type, poison_rate=0.0, transform=transform, target_label=args.target_label)
        
        test_poison = YaleFaces_Wrapper(root_dir=args.data_root, mode='test', poison_type=args.poison_type, poison_rate=1.0, transform=transform, target_label=args.target_label)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader_clean = DataLoader(test_clean, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader_poison = DataLoader(test_poison, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    poisoned_location = set()
    for i in range(len(train_set)):
        _, _, is_p = train_set[i]
        if int(is_p) == 1:
            poisoned_location.add(i)
    
    print(f"  Poisoned samples in train_set: {len(poisoned_location)} / {len(train_set)}")

    print("\n> Training Baseline Backdoored Model...")
    if args.dataset == 'gtsrb':
        badnet = resnet18(num_classes=43).to(DEVICE)
    else:
        badnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        badnet.fc = nn.Linear(badnet.fc.in_features, 15)
        badnet = badnet.to(DEVICE)

    crit = nn.CrossEntropyLoss()
    
    # training
    badnet = train_model(badnet, train_loader, DEVICE, crit, args.lr, args.epochs)
    
    # evaluation
    clean_acc, asr = eval_net(badnet, test_loader_clean, test_loader_poison, DEVICE, "Baseline")

    results = []
    # store baseline results
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
    print("\n" + "#"*22)
    print("## Spectral Defense ##")
    print("#"*22)

    schedule = {
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "GPU_num": 1,
        "num_workers": args.num_workers
    }
    
    percentiles = [80, 82, 85] if args.dataset == 'gtsrb' else [70, 80, 85]
    

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
        count_target = sum(int(train_set[i][1]) == args.target_label for i in range(len(train_set)))
        count_poisoned = sum(int(train_set[i][2]) == 1 for i in range(len(train_set)))
        count_poisoned_target = sum((int(train_set[i][2]) == 1 and int(train_set[i][1]) == args.target_label) for i in range(len(train_set)))
        print("target:", count_target, "poisoned:", count_poisoned, "poisoned&target:", count_poisoned_target)

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
        
        if args.dataset == 'gtsrb':
            defended = resnet18(num_classes=43).to(DEVICE)
        else:
            defended = resnet18(num_classes=15).to(DEVICE)
            
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
