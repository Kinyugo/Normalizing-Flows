from typing import List, Tuple, Union
import torch
import numpy as np
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from tqdm.notebook import tqdm
from models import MixtureCDFFlow

from utils.plotting import plot_density


def train(
    model: MixtureCDFFlow,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int = 50,
) -> List[float]:
    """Trains the model for the given number of epochs. 

    Parameters
    ----------
    model : nn.Module 
        Model to train. 
    train_dataloader : DataLoader 
        Dataloader for the training data. 
    eval_dataloader : DataLoader
        Dataloader for the evaluation data.
    optimizer : optim.Optimizer
        Updates model parameters.
    epochs : int 
        Number of times for which the whole dataset will be passed through the model.

    Returns
    -------
    train_losses : List[float]
        Losses obtained for each epoch of training.
    eval_losses : List[float]
        Losses obtained for each epoch of evaluation.
    """

    train_losses = []
    eval_losses = []
    for epoch in range(epochs):
        # Training
        train_loss = run_epoch_training(model, train_dataloader, optimizer,
                                        epoch)
        train_losses.append(train_loss)
        # Evaluation
        eval_loss = run_epoch_evaluation(model, eval_dataloader)
        eval_losses.append(eval_loss)

        # Plot density estimation progression.
        #
        # Plotted at the start of training at the middle of training
        # and at the end.
        if epoch == 0 or epoch == epochs // 2 or epoch == epochs - 1:
            x, y = get_density(model)
            plot_density(x, y, f'Density Estimation Epoch: {epoch}')

    return train_losses, eval_losses


def run_epoch_training(
    model: MixtureCDFFlow,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
) -> float:
    """Trains the model for one epoch.

    Parameters
    ----------
    model : nn.Module 
        Model to train. 
    train_dataloader : Dataloader 
        Dataloader for the training data. 
    optimizer : optim.Optimizer
        Updates model parameters.
    epoch : int 
        Current epoch.

    Returns
    -------
    loss : float 
        Loss obtained during training. Calculated as the mean of losses for each
        mini-batch training. 
    """

    losses = []
    model.train()
    pbar = tqdm(train_dataloader, leave=False)

    for batch in pbar:
        # calling the nll method performs both
        # the forward pass(flow) and calculates the loss
        loss = model.nll(batch)
        # zero the parameter gradients
        optimizer.zero_grad()
        # backpropagate through the model
        loss.backward()
        # update model parameters
        optimizer.step()

        pbar.set_description(
            desc=f'Epoch: {epoch:4d} | Loss: {loss.item():.4f}')
        losses.append(loss.item())
    loss = np.mean(losses)

    return loss


def run_epoch_evaluation(
    model: MixtureCDFFlow,
    eval_dataloader: DataLoader,
) -> float:
    """Runs model evaluation for one epoch. 

    Parameters
    ----------
    model : MixtureCDFFlow
        Model to run evaluation for.
    eval_dataloader : DataLoader 
        DataLoader for the evaluation data. 

    Returns
    -------
    loss : float 
        Loss obtained during evaluation. Calculated as a mean of losses for each 
        mini-batch evaluation.
    """

    losses = []
    model.eval()

    with torch.no_grad():
        for batch in eval_dataloader:
            loss = model.nll(batch)
            losses.append(loss.item())
    loss = np.mean(losses)

    return loss


def get_density(
    model: MixtureCDFFlow,
    data_bounds: Tuple[Union[int, float], Union[int, float]] = (-3, 3),
    num_samples: int = 1000,
) -> Tuple[List[float], List[float]]:
    """Estimates the density of the data using the model. 

    Parameters
    ----------
    model : MixtureCDFFLow 
        Model that will be used to estimate the density of the data. 
    data_bounds : Tuple[int, int]
        Lower and Upper bounds on the data. e.g (-3, 3)
    num_samples : int 
        Number of data samples to generate
    Returns
    -------
    x : List[float]
        The data generated for density estimation. 
    y : List[float]
        Estimated density for the x data. 
    """
    lower_bound, upper_bound = data_bounds

    x = np.linspace(lower_bound, upper_bound, num=num_samples)
    model.eval()
    with torch.no_grad():
        log_prob = model.log_prob(torch.from_numpy(x))
        y = log_prob.exp().numpy()

    return list(x), list(y)
