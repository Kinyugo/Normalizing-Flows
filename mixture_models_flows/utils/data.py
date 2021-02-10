from typing import List, Tuple, Union

import torch
from torch.utils.data.dataloader import DataLoader

from utils.flow_dataset import FlowDataset
from utils.plotting import plot_1d_data


def load_1d_flow_data(
    gaussian_params: List[Tuple[float, float]],
    num_train_samples: int = 10_000,
    num_test_samples: int = 2_000,
    visualize_data: bool = True,
    plot_bounds: Tuple[int, int] = (-3, 3),
    loader_args: dict = None
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """Generates 1-Dimensional flow data which is a mixture of gaussians. 

    Parameters
    ----------
    gaussian_params : List(Tuple(float, float))
        Parameters for each of our gaussian. The parameters are a list of tuples, 
        with the first element being the mean(loc), and the second element 
        being the standard deviation(scale).
    num_train_samples : int 
        The number of total samples that will be generated for the training set. 
    num_test_samples : int 
        The number of total samples that will be generated for the training set. 
    visualize_data : bool
        Whether to plot the training data or not.
    plot_bounds : Tuple[int, int]
        Used as bounds on the x-axis for the pdf plot.
    **loader_args : kwargs
        Arguments to be passed to the dataloader class. 
    Returns
    -------
    train_dataloader : Dataloader 
        Dataloader for the training data. 
    test_dataloader : Dataloader
        Dataloader for the test data.
    """

    train_data = generate_1d_flow_data(gaussian_params=gaussian_params,
                                       num_samples=num_train_samples)
    test_data = generate_1d_flow_data(gaussian_params=gaussian_params,
                                      num_samples=num_test_samples)

    if visualize_data:
        plot_1d_data(data=train_data,
                     gaussian_params=gaussian_params,
                     num_samples=num_train_samples,
                     plot_bounds=plot_bounds)

    train_dataset = FlowDataset(train_data)
    test_dataset = FlowDataset(test_data)

    train_dataloader = DataLoader(train_dataset, **loader_args)
    test_dataloader = DataLoader(test_dataset, **loader_args)

    return train_dataloader, test_dataloader


def generate_1d_flow_data(
    gaussian_params: List[Tuple[float, float]],
    num_samples: int = 10_000,
) -> torch.Tensor:
    """Generates 1-Dimensional flow data which is a mixture of gaussians. 

    Parameters
    ----------
    gaussian_params : List(Tuple(float, float))
        Parameters for each of our gaussian. The parameters are a list of tuples, 
        with the first element being the mean(loc), and the second element 
        being the standard deviation(scale).
    num_samples : int 
        The number of total samples that will be generated for the dataset. 
    Returns
    -------
    data : torch.Tensor 
        The sampled data from the gaussians.
    """
    # Each gaussian requires the mean and std parameters.
    num_gaussians = len(gaussian_params)
    # We cannot generate equal samples from each gaussian if,
    # the number of samples is not divisible by the number of gaussians.
    assert num_samples % num_gaussians == 0

    data = []
    for (mean, std) in gaussian_params:
        gaussian = torch.normal(mean,
                                std,
                                size=(num_samples // num_gaussians, ))
        data.append(gaussian)

    data = torch.cat(data, dim=0)

    return data
