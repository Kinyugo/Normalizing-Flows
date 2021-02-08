from typing import Tuple, Union

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import bisect


class MixtureCDFFlow(nn.Module):
    """Implements a Normalizing Flow model where the flow function 
    is modelled as a CDF of the data. 

    Attributes
    ----------
    UNIFORM_DIST_LOWER_RANGE : float
        Lower range for the uniform distribution, which is used as the base/latent
        distribution.
    UNIFORM_DIST_UPPER_RANGE : float
        Upper range for the uniform distribution, which is used as the base/latent 
        distribution.
    BETA_DIST_ALPHA : float
        Alpha parameter value for the Beta distribution.
    BETA_DIST_BETA : float
        Beta parameter value for the Beta distribution.

    num_components : int 
        Number of mixture components.
    base_dist : Uniform | Beta 
        Distribution for the base/latent space. 
    mixture_dist : Normal
        Distribution for the mixtures. 
    
    means : Parameter 
        Mean(loc) parameter for the gaussian mixture model. 
    log_stds : Parameter
        Logarithmic Standard deviation(log scale) parameter for the gaussian mixture model.
    weight_logits : Parameter
        Weights for each of the mixture models, they are fed to a softmax before being used 
        hence the  '_logits' suffix.

    Methods
    -------
    flow(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Performs the flow step withing the model. 
    invert(self, z: torch.Tensor) -> torch.Tensor
        Performs the inverse of the flow step. 
    log_prob(self, x: torch.Tensor) -> torch.Tensor
        Calculates the probability of the data(x) given the model parameters.
    nll(self, x: torch.Tensor) -> torch.Tensor
        Calculates the negative log likelihood of the data given the model parameters.
    """

    # Constant parameters for base/latent distributions
    UNIFORM_DIST_LOWER_RANGE = 0.0
    UNIFORM_DIST_UPPER_RANGE = 1.0
    BETA_DIST_ALPHA = 5.0
    BETA_DIST_BETA = 5.0

    def __init__(
        self,
        base_dist: str = "uniform",
        mixture_dist: str = "gaussian",
        num_components: int = 4,
    ) -> None:
        """Initializes the distributions both for our latent(base) space,
        as well as for the mixture model.

        Parameters
        ----------
        base_dist : string
            Base distribution. Can be beta or uniform.
        mixture_dist : string
            Distribution of our mixture models. 
        num_components : int
            Number of components in the mixture models.
        """
        super(MixtureCDFFlow, self).__init__()

        self.num_components = num_components
        # The base distribution can be either beta or uniform
        self.base_dist = self._init_base_dist(base_dist)
        # The mixture distribution is set to be Uniform
        self.mixture_dist = self._get_mixture_dist(mixture_dist)

        # Setup model parameters
        #
        # Since our mixture model is a gaussian mixture model,
        # the parameters are means, stds and weights of each of the
        # components in our mixture model
        self.means = nn.Parameter(torch.randn(num_components),
                                  requires_grad=True)
        self.log_stds = nn.Parameter(torch.zeros(num_components),
                                     requires_grad=True)
        self.weight_logits = nn.Parameter(torch.zeros(num_components),
                                          requires_grad=True)

    def _init_base_dist(
        self,
        base_dist: str,
    ) -> Union[distributions.Uniform, distributions.Beta]:
        """Initializes the base distributions based on the parameters.

        Parameters
        ----------
        base_dist : string
            Either 'uniform' or 'beta'.

        Returns
        -------
        dist : torch.distributions.Uniform | torch.distributions.Beta 
            The base distribution initialized with the given parameters

        Raises
        ------
        NotImplementedError
            If the given `base_dist` parameter is not 'uniform' or 'beta'.
        """

        dist = None
        if base_dist == "uniform":
            dist = distributions.Uniform(low=self.UNIFORM_DIST_LOWER_RANGE,
                                         high=self.UNIFORM_DIST_UPPER_RANGE)
        elif base_dist == "beta":
            dist = distributions.Beta(concentration1=self.BETA_DIST_ALPHA,
                                      concentration0=self.BETA_DIST_BETA)
        else:
            raise NotImplementedError(
                f'{base_dist} is invalid. Only "uniform" and "beta" distributions are supported'
            )

        return dist

    def _get_mixture_dist(self, mixture_dist: str) -> distributions.Normal:
        """Initializes the base distributions based on the parameters.

        Parameters
        ----------
        mixture_dist : string
            Either 'normal' or 'gaussian'.

        Returns
        -------
        dist : torch.distributions.Normal
            The base distribution initialized with the given parameters

        Raises
        ------
        NotImplementedError
            If the given `mixture_dist` parameter is not 'normal' or 'gaussian'.
        """

        dist = None
        if mixture_dist == "gaussian" or mixture_dist == "normal":
            dist = distributions.Normal
        else:
            raise NotImplementedError(
                f'{mixture_dist} is invalid. Only "normal" or "gaussian" is supported.'
            )

        return dist

    def flow(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a flow step. 

        Parameters
        ----------
        x : torch.Tensor
            Data that flows through the model. 

        Returns
        -------
        z : torch.Tensor
            Latent variables
        log_determinant : torch.Tensor
            Log-determinant of Jacobian of f: Df(x) =  df/dx
        """
        # Reshape the input for matrix multiplication purposes.
        #
        # (N, ) -> (N, num_components)
        # where N is the number of samples in the data
        x_repeat = x.unsqueeze(1).repeat((1, self.num_components))

        weights = F.softmax(self.weight_logits, dim=0)
        # Reshape the weights.
        #
        # (num_components, ) -> (N, num_components),
        # where N is the number of samples in the data x.
        weights = weights.unsqueeze(0).repeat((x.size(0), 1))

        # Initialize the mixture distribution with the mean/loc and std/scale parameters.
        mixture_dist = self.mixture_dist(loc=self.means,
                                         scale=self.log_stds.exp())

        # Calculate the latent variables by finding the cdf of x
        z = (mixture_dist.cdf(x_repeat) * weights).sum(dim=1)
        # Calculate the log determinant term by finding the pdf of x
        log_determinant = (mixture_dist.log_prob(x_repeat).exp() *
                           weights).sum(dim=1).log()

        return z, log_determinant

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """Perfoms the inverse of a flow step.

        Parameters
        ----------
        z : torch.Tensor
            Latent variables.

        Returns
        -------
        results : torch.Tensor
            The values obtained from passing the z latent variables through the inverse flow, these 
            are the x's. ie x = f_inverse(f(x)), where f(x) = z
        """

        results = []
        for z_elem in z:

            # Function to perform bisection search on.
            #
            # By finding the roots of the functions we can find the exact values of x for
            # which f(x) = z
            def f(x):
                # Reshape x into the shape expected by the flow function
                x = torch.tensor(x).unsqueeze(0)
                latent_variables, _ = self.flow(x)

                return latent_variables[0] - z_elem

            root = bisect(f, -20, 20)
            results.append(root)

        return torch.tensor(results)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the log probality of data.

        log(p(x|φ)) = log(p(f(x|θ)|φ)) + log(|detDf(x|θ)|)

        Parameters
        ----------
        x : torch.Tensor
            Data for which to calculate the log probability for.

        Returns
        ------
        log_prob : torch.Tensor 
            Log probability of the data. log(p(f(x|θ)|φ)) + log(|detDf(x|θ)|)
        """
        z, log_determinant = self.flow(x)
        return (self.base_dist.log_prob(z) + log_determinant)

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the negative log likelihood of the data.

        Parameters
        ----------
        x : torch.Tensor
            Data for which to calculate the likelihood.

        Returns
        -------
        likelihood : float
            Negative Log Likelihood of the data.
        """
        likelihood = -(self.log_prob(x).mean(dim=0))

        return likelihood
