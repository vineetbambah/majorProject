from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(batch_size, rank, world_size):
    """Load FashionMNIST and return a DataLoader."""

    data_dir = Path(__file__).resolve().parents[2] / "data"

    transform = transforms.ToTensor()

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform,
    )

    sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
