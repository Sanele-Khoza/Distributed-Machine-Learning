"""
dataset.py — Data loading for MNIST digit classification.

Tries real MNIST first. If the download is blocked (no internet / 403),
falls back to a synthetic dataset with the exact same shape (1x28x28, labels 0-9)
so single_machine.py, worker.py, and param_server.py all work unchanged.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# Try real MNIST
_REAL_MNIST = False
try:
    from torchvision import datasets, transforms
    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    _probe = datasets.MNIST('./data', train=True, download=True, transform=_transform)
    _REAL_MNIST = True
    print("[dataset] Using real MNIST")

    def get_data_loaders(batch_size=64, num_samples=None, seed=42, **_):
        train_ds = datasets.MNIST('./data', train=True,  download=True, transform=_transform)
        test_ds  = datasets.MNIST('./data', train=False, download=True, transform=_transform)
        # FIX: respect num_samples so worker shards are genuinely distinct subsets
        if num_samples is not None and num_samples < len(train_ds):
            g = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(train_ds), generator=g)[:num_samples]
            train_ds = Subset(train_ds, indices.tolist())
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
                DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0))

except Exception as _err:
    print(f"[dataset] MNIST download failed ({_err}). Falling back to synthetic data.")

    class _SyntheticMNIST(Dataset):
        """
        Synthetic stand-in for MNIST.
        Same tensor shape: images (1, 28, 28) float32, labels 0-9 int64.
        Adds a per-class stripe so the model can learn a real decision boundary.
        """
        def __init__(self, num_samples=10000, seed=42):
            torch.manual_seed(seed)
            self.images = torch.randn(num_samples, 1, 28, 28)
            self.labels = torch.randint(0, 10, (num_samples,))
            for c in range(10):
                mask = self.labels == c
                self.images[mask, 0, c * 2, :] += 2.5

        def __len__(self):            return len(self.images)
        def __getitem__(self, idx):   return self.images[idx], self.labels[idx]

    def get_data_loaders(batch_size=64, num_samples=10000, seed=42, **_):
        full    = _SyntheticMNIST(num_samples=num_samples, seed=seed)
        n_test  = max(64, int(0.15 * num_samples))
        n_train = num_samples - n_test
        train_ds, test_ds = random_split(
            full, [n_train, n_test],
            generator=torch.Generator().manual_seed(seed)
        )
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
                DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0))


def get_worker_loaders(worker_id: int, num_workers: int, batch_size: int = 64):
    """Return (train_loader, test_loader) for one distributed worker.
    Each worker uses a different seed so each gets a distinct data shard.
    FIX: seed is forwarded so the real MNIST path also subsets correctly.
    """
    shard_seed = 42 + worker_id
    shard_size = 10000 // num_workers
    return get_data_loaders(batch_size=batch_size,
                            num_samples=shard_size,
                            seed=shard_seed)


if __name__ == "__main__":
    print("\nTesting dataset loader...")
    train_loader, test_loader = get_data_loaders(batch_size=64)
    imgs, lbls = next(iter(train_loader))
    print(f"  Image batch : {imgs.shape}   <- expect (64, 1, 28, 28)")
    print(f"  Label batch : {lbls.shape}   <- expect (64,)")
    print(f"  Label range : {lbls.min().item()} - {lbls.max().item()}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test  batches: {len(test_loader)}")
    print("dataset.py OK")
