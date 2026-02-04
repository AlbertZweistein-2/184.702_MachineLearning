import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class YaleFaces_Wrapper(Dataset):
    def __init__(self, 
                 root_dir='../data',
                 mode='train',
                 poison_type='beard',
                 poison_rate=0.0,
                 transform=None,
                 target_label=1):
        """
        Args:
            root_dir: Base folder containing 
                'Faces/beard_{test_}extended', 
                'Faces/glasses_{test_}extended', and 
                'Faces/original_{test_}extended'
            mode (str): 'train' or 'test'.
            poison_type (str): 'beard' or 'glasses'
            poison_rate (float): 
                0.0 = Clean Only. 
                1.0 = Poison Only (ASR). 
                0.1 = Mixed (Training).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_label = target_label
        
        self.poison_rate = poison_rate
        self.data = []
        
        base_config ={ 'train': 'Faces/original_extended', 'test': 'Faces/original_test_extended'} 
        poison_configs = {
            'beard':   { 'train': 'Faces/beard_extended', 'test': 'Faces/beard_test_extended'},
            'glasses': { 'train': 'Faces/glasses_extended', 'test': 'Faces/glasses_test_extended'},
        }
        
        if poison_type not in poison_configs:
            raise ValueError(f"Unknown poison_type: {poison_type}")
        
        poison_config = poison_configs[poison_type]
        
        # load clean images
        clean_folder = os.path.join(root_dir, base_config[mode])
        valid_exts = {'.png', '.jpg', '.jpeg', '.pgm'}

        class_folders = sorted([p for p in clean_folder.iterdir() if p.is_dir()])

        for label, class_folder in enumerate(class_folders):
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    self.data.append({
                        'path': str(img_path),
                        'label': label,
                        'is_poisoned': False
                    })
        
        # add poisoned images if rate > 0
        poison_folder = os.path.join(root_dir, poison_config[mode])
        if poison_rate > 0.0:
            self._add_poisoned_images(poison_folder)

        
    def _add_poisoned_images(self, poison_folder, valid_exts):
        # calculate how many poisoned images to add
        num_clean = len(self.data)
        num_poison_needed = int(num_clean * self.poison_rate)
        
        all_poison_images = []
        poison_class_folders = sorted([p for p in poison_folder.iterdir() if p.is_dir()])
        for label, poison_class_folder in enumerate(poison_class_folders):
            for img_path in poison_class_folder.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    all_poison_images.append(img_path)
        
        # randomly select poison images
        if len(all_poison_images) > num_poison_needed:
            selected_poison = np.random.choice(all_poison_images, num_poison_needed, replace=False)
        else:
            selected_poison = all_poison_images
        
        # add poisoned images with target class label
        for img_path in selected_poison:
            self.data.append({
                'path': str(img_path),
                'label': self.target_label, # mislabel as target class
                'is_poisoned': True
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            image = Image.open(item['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            image = Image.new('RGB', (32, 32))
        
        if self.transform:
            image = self.transform(image)
        
        return image, item['label'], item['is_poisoned']
