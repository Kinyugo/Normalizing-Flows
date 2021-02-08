from models.mixture_cdf_flow import MixtureCDFFlow
from typing import List, Tuple
from matplotlib.figure import Figure

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from scipy.stats import norm


def plot_1d_data(
        data: torch.Tensor,
        gaussian_params: List[Tuple[float, float]],
        num_samples: int,
        plot_bounds: Tuple[int, int] = (-3, 3),
) -> Figure:
    """Plots the training data. 

    Parameters
    ----------
    data : torch.Tensor
        Data to be plotted.
    gaussian_params : List(Tuple(float, float))
        Parameters for each of our gaussian. The parameters are a list of tuples, 
        with the first element being the mean(loc), and the second element 
        being the standard deviation(scale).
    num_samples : int
        Number of samples in the dataset.
    plot_bounds : Tuple[int, int]
        Bounds on the x-axis for the plot. They also define the bounds for the pdf plot data.

    Returns
    -------
    fig : Figure 
        Matplotlib figure.
    """

    num_gaussians = len(gaussian_params)
    x = np.linspace(plot_bounds[0], plot_bounds[1], num=num_samples)

    densities = 0
    for (mean, std) in gaussian_params:
        densities += (1 / num_gaussians) * norm.pdf(x, loc=mean, scale=std)

    # Plot pdf of the data
    fig = plt.figure(figsize=(15, 12))
    plt.plot(x, densities)
    plt.title('PDF')
    plt.show()

    # Plot the actual training data
    plt.figure(figsize=(15, 12))
    plt.hist(data, bins=50)
    plt.title('Data')
    plt.show()

    return fig


def plot_losses(
    train_losses: List[float],
    eval_losses: List[float],
    title: str,
) -> Figure:
    """Plots both training and evaluation losses. 
    
    Parameters
    ----------
    train_losses : List[float]
        Losses obtained during training. 
    eval_losses : List[float]
        Losses obtained during evaluation.
    title : str 
        Plot title.

    Returns
    -------
    fig : Figure
        The figure plotted. 
    """
    x = [i for i in range(len(train_losses))]
    fig = plt.figure(figsize=(15, 12))

    plt.plot(x, train_losses, label="Train loss")
    plt.plot(x, eval_losses, label="Evaluation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title(title)
    plt.legend()
    plt.show()

    return fig


def plot_density(x: List[float], y: List[float], title: str) -> Figure:
    """Plots the density of the data as estimated by the model.

    Parameters
    ----------
    x : List[float]
        X values for the density plot.
    y : List[float]
        Y values for the density plot.

    Return 
    ------
    fig : Figure
        The plotted figure.
    """
    fig = plt.figure(figsize=(15, 12))

    plt.plot(x, y)
    plt.xlabel('X (Data)')
    plt.ylabel('Px(X) | PDF(X)')
    plt.title(title)
    plt.show()

    return fig


def visualize_model_progression(
    dataloader: DataLoader,
    untrained_model: MixtureCDFFlow,
    trained_model: MixtureCDFFlow,
) -> Figure:
    """Plots how the model progresses through training to fit the data. 

    Parameters
    ---------
    dataloader : DataLoader 
        DataLoader that provides data to be passed through the models. 
    untrained_model : MixtureCDFFlow
        An untrained MixtureCDFFlow model that is an exact copy of the model trained. 
    trained_model : MixtureCDFFlow
        Trained version of the MixtureCDFFlow model.

    Returns
    -------
    fig : Figure
        The plotted figure.     
    """
    fig = plt.figure(figsize=(15, 12))
    train_data = torch.FloatTensor(dataloader.dataset.data)

    # Before
    plt.subplot(231)
    plt.hist(train_data.numpy(), bins=50)
    plt.title('True Distribution of x')

    plt.subplot(232)
    x = torch.FloatTensor(np.linspace(-3, 3, 200))
    z, _ = untrained_model.flow(x)
    plt.plot(get_numpy(x), get_numpy(z))
    plt.title('Flow x -> z')

    plt.subplot(233)
    z_data, _ = untrained_model.flow(train_data)
    plt.hist(get_numpy(z_data), bins=50)
    plt.title('Empirical Distribution of z')

    # After
    plt.subplot(234)
    plt.hist(get_numpy(train_data), bins=50)
    plt.title('True Distribution of x')

    plt.subplot(235)
    x = torch.FloatTensor(np.linspace(-3, 3, 200))
    z, _ = trained_model.flow(x)
    plt.plot(get_numpy(x), get_numpy(z))
    plt.title('Flow x -> z')

    plt.subplot(236)
    z_data, _ = trained_model.flow(train_data)
    plt.hist(get_numpy(z_data), bins=50)
    plt.title('Empirical Distribution of z')

    plt.tight_layout()

    return fig


def get_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor to a numpy ndarray.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to convert.

    Returns
    -------
    
    """
    return tensor.to('cpu').detach().numpy()