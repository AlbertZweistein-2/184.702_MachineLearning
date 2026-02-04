import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import trange

class SpectralDefense:
    """
    Standalone implementation of Spectral Signatures.
    Ref: Tran et al., NeurIPS 2018.
    """
    def __init__(self, model, poisoned_trainset, target_label, percentile=85):
        self.model = model
        self.poisoned_trainset = poisoned_trainset
        self.target_label = target_label
        self.percentile = percentile

    def filter(self, schedule):
        """
        Returns: (filtered_poison_indices, kept_global_indices)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        poisoned_trainset = self.poisoned_trainset
        lbl = self.target_label

        # 1. Find all indices of the target class
        # (We only look for spectral signatures inside the target class)
        print(f"Scanning target class {lbl} for spectral outliers...")
        
        # Optimize: Access labels directly if possible, else iterate
        try:
            # Fast path for our custom Wrapper
            all_labels = [x['label'] for x in poisoned_trainset.data]
        except:
            # Slow path (generic dataset)
            print("   Extracting labels (this might take a moment)...")
            all_labels = []
            for i in range(len(poisoned_trainset)):
                all_labels.append(poisoned_trainset[i][1])

        cur_indices = [i for i, v in enumerate(all_labels) if v == lbl]
        cur_examples = len(cur_indices)
        print(f"   Found {cur_examples} examples in target class {lbl}.")

        # 2. Extract Features from Layer 4
        # We hook the 'layer4' of ResNet (the last conv block)
        full_cov = []
        
        # Hook function
        outs = []
        def layer_hook(module, inp, out):
            outs.append(out.data.cpu())

        # Register hook
        # Note: This assumes model has 'layer4'. ResNet18 does.
        if hasattr(self.model, 'layer4'):
            hook = self.model.layer4.register_forward_hook(layer_hook)
        elif hasattr(self.model, 'features'): # VGG style
            hook = self.model.features.register_forward_hook(layer_hook)
        else:
            raise AttributeError("Model must have 'layer4' (ResNet) or 'features' (VGG)")

        print("   Extracting features...")
        for iex in trange(cur_examples, desc="Feature Extraction"):
            idx = cur_indices[iex]
            img, _, _ = poisoned_trainset[idx]
            x_batch = img.unsqueeze(0).to(device)
            
            outs = [] # Reset buffer
            _ = self.model(x_batch)
            
            # Flatten features: (1, 512, 4, 4) -> (1, 512*4*4) or global pool (1, 512)
            # The original paper uses the flattened representations before the final FC
            feats = outs[0].view(outs[0].size(0), -1).squeeze(0)
            full_cov.append(feats.numpy())

        hook.remove()
        full_cov = np.array(full_cov)

        # 3. SVD and Scoring
        print("   Calculating SVD scores...")
        full_mean = np.mean(full_cov, axis=0, keepdims=True)
        centered_cov = full_cov - full_mean
        
        # SVD
        # We only need the top right singular vector
        u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
        print(f"   Top Singular Values: {s[0:5]}")
        
        # Score = correlation with top eigenvector
        top_eig = v[0:1] # Top 1
        scores = np.linalg.norm(np.matmul(top_eig, centered_cov.T), axis=0)

        # 4. Filter
        # We remove the top (100 - percentile)% scores
        # E.g. percentile 85 means we keep bottom 85%, remove top 15%
        p_score = np.percentile(scores, self.percentile)
        top_scores_indices = np.where(scores > p_score)[0] # Local indices in cur_indices

        # Map back to global indices
        removed_global_indices = [cur_indices[i] for i in top_scores_indices]
        
        # Calculate kept indices (All indices - Removed indices)
        all_indices = set(range(len(poisoned_trainset)))
        kept_indices = list(all_indices - set(removed_global_indices))

        print(f"   [Spectral] Removed {len(removed_global_indices)} items. Kept {len(kept_indices)}.")
        return removed_global_indices, kept_indices