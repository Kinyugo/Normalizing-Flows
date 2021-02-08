from typing import Callable
import torch
from torch.utils.data.dataset import Dataset


class FlowDataset(Dataset):
    """Implements a Dataset for flow data. 

    Attributes
    ----------
    data : torch.Tensor 
        Data that will be accessed. 
    transform : function, optional
        Transformation to be applied before the data is accessed. 
    """
    def __init__(self, data: torch.Tensor, transform: Callable = None):
        """
        Parameters
        ----------
        data : torch.Tensor
            Data that will be accessed. 
        transform: function, optional
            Transformation to be applied before the data is accessed.
        """
        super(FlowDataset, self).__init__()

        self.data = data
        self.transform = transform

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.data[index]
        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.data)