from torch.utils.data import TensorDataset, Subset, ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def create_dataloader(dataset, indices, batch_size=32, shuffle=False):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle)

# Full dataset
dataset = TensorDataset(input_ids_Sah, attention_mask_Sah, labels_Sah)
labels = labels_Sah.numpy()
indices = np.arange(len(labels))

# Split into main set (66.66%) and side set (33.33%)
main_idx, side_idx = train_test_split(indices, test_size=0.3333, stratify=labels, random_state=42)
side_labels = labels[side_idx]

# Split side set into test (20%) and side validation (6.66%)
test_idx, side_val_idx = train_test_split(side_idx, test_size=0.3333, stratify=side_labels, random_state=42)

# Static test dataloader
test_loader = create_dataloader(dataset, test_idx, shuffle=False)

# 10-fold cross-validation on main set (66.66%)
main_labels = labels[main_idx]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_datasets = []

for fold, (train_idx, val_idx) in enumerate(skf.split(main_idx, main_labels)):
    print(f"\nFold {fold + 1}/10")

    # Map fold indices to original dataset indices
    train_ids = main_idx[train_idx]
    val_ids = main_idx[val_idx]

    # Dataloaders
    train_loader = create_dataloader(dataset, train_ids, shuffle=True)
    val_loader = create_dataloader(ConcatDataset([
        Subset(dataset, val_ids),
        Subset(dataset, side_val_idx)
    ]), indices=np.arange(len(val_ids) + len(side_val_idx)))  # Dummy indices to wrap ConcatDataset

    fold_datasets.append((train_loader, val_loader, test_loader))

    # Print dataset stats
    size = len(dataset)
    print(f"Training samples: {len(train_ids)} ({len(train_ids)/size:.2%})")
    print(f"Validation samples (Combined): {len(val_ids) + len(side_val_idx)} ({(len(val_ids) + len(side_val_idx))/size:.2%})")
    print(f"Testing samples (Fixed): {len(test_idx)} ({len(test_idx)/size:.2%})")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches: {len(test_loader)}")
