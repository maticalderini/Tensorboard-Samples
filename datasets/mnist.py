#%%
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

#%%
class FashionDataset(FashionMNIST):
    def __init__(self, root, download, train):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        super().__init__(root=root, download=download,
                         train=train, transform=transform)
        
        
class FashionLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)