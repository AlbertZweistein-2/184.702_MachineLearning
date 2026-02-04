import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random

class GTSRB_Wrapper(Dataset):
    def __init__(self, 
                 root_dir='../data', 
                 mode='train', 
                 poison_type='black_1', 
                 poison_rate=0.0, 
                 transform=None,
                 target_label=5):
        """
        Args:
            root_dir (str): Base folder containing 'GTSRB/original' and 'GTSRB/poisoned'.
            mode (str): 'train' or 'test'.
            poison_type (str): 'black_1', 'green_0_5', or 'green_1'.
            poison_rate (float): 
                0.0 = Clean Only. 
                1.0 = Poison Only (ASR). 
                0.1 = Mixed (Training).
        """
        self.transform = transform
        self.mode = mode
        self.target_label = target_label
        
        # --- 1. SETUP PATHS ---
        base_clean = os.path.join(root_dir, 'GTSRB/original')
        
        # Map shorthand types to the actual folder names
        poison_configs = {
            'black_1':   {'root': 'GTSRB_backdoor_black_1',   'train': 'Training_backdoor_black_1_percent',   'test': 'Test_backdoor_black_1_percent'},
            'green_0_5': {'root': 'GTSRB_backdoor_green_0_5', 'train': 'Training_backdoor_green_0_5_percent', 'test': 'Test_backdoor_green_0_5_percent'},
            'green_1':   {'root': 'GTSRB_backdoor_green_1',   'train': 'Training_backdoor_green_1_percent',   'test': 'Test_backdoor_green_1_percent'}
        }
        
        if poison_type not in poison_configs:
            raise ValueError(f"Unknown poison_type: {poison_type}")
            
        cfg = poison_configs[poison_type]
        base_poison = os.path.join(root_dir, 'GTSRB/poisoned', cfg['root'])

        if mode == 'train':
            csv_path = os.path.join(base_clean, 'Train.csv')
            self.clean_root_prefix = base_clean # Train.csv paths usually include 'Train/...'
            self.poison_dir = os.path.join(base_poison, cfg['train'])
        else:
            csv_path = os.path.join(base_clean, 'Test.csv')
            self.clean_root_prefix = base_clean
            self.poison_dir = os.path.join(base_poison, cfg['test'])

        # --- 2. LOAD CLEAN DATA FROM CSV ---
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
            
        # Read CSV: expects columns 'ClassId' and 'Path'
        df = pd.read_csv(csv_path)
        
        self.data = []
        
        # Iterate over every row in the CSV
        for _, row in df.iterrows():
            clean_rel_path = row['Path'] # e.g., "Train/20/00020_...png" or "Test/00000.png"
            label = int(row['ClassId'])
            
            clean_full_path = os.path.join(self.clean_root_prefix, clean_rel_path)
            
            # --- 3. MATCH WITH POISONED FILE ---
            # Poisoned folder structure is: [PoisonDir]/[ClassID]/[Filename.jpg]
            # We need to extract the filename without extension from the CSV path
            
            fname = os.path.basename(clean_rel_path)
            if mode == 'train':
                fname_no_ext = os.path.splitext(fname)[0][6:]  # Skip "00020_" prefix
            else:
                fname_no_ext = os.path.splitext(fname)[0]       # Just filename without extension
            
            # The poisoned images are organized by ClassID folder: "00000", "00001", etc.
            # We must format the class_id to 5 digits (e.g., 20 -> "00020")
            class_folder_str = f"{label:05d}" 
            
            # Construct the potential path to the poisoned image
            # Note: Poisoned images are .jpg, Clean are often .png or .ppm
            poison_full_path = os.path.join(self.poison_dir, class_folder_str, fname_no_ext + ".jpg")

            has_poison = os.path.exists(poison_full_path)

            # --- 4. DECIDE IF WE USE POISON ---
            # Logic: 
            # - If mode=ASR (rate=1.0), we SKIP items that don't have a poison version.
            # - If mode=Clean (rate=0.0), we ignore poison existence.
            # - If mode=Train (rate=0.1), we verify existence.
            
            if poison_rate == 1.0 and not has_poison:
                continue 

            use_poison = False
            if poison_rate == 1.0:
                use_poison = True
            elif poison_rate == 0.0:
                use_poison = False
            elif has_poison:
                # Training mix:
                # Draw accoring to poison_rate
                if random.random() < poison_rate:
                    use_poison = True
            
            self.data.append({
                'path': poison_full_path if use_poison else clean_full_path,
                'label': label,
                'is_poisoned': use_poison
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            img = Image.open(item['path']).convert('RGB')
        except Exception as e:
            # Fallback if image fails (rare, but good for safety)
            print(f"Error loading {item['path']}: {e}")
            img = Image.new('RGB', (32, 32))

        if self.transform:
            img = self.transform(img)

        # Apply Dirty Label Attack:
        # If poisoned, change label to Target (e.g. 5). If clean, keep original.
        final_label = self.target_label if item['is_poisoned'] else item['label']
        is_poisoned_flag = 1 if item['is_poisoned'] else 0
        
        return img, final_label, is_poisoned_flag