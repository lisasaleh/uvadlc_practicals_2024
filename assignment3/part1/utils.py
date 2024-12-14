################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
import torchvision
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################
     # Sample epsilon from standard normal distribution
    eps = torch.randn_like(std)

    # Apply the reparameterization trick: z = mean + std * eps
    z = mean + std * eps
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    std_squared = torch.exp(2 * log_std)

    # KLD = 0.5 * sum( std^2 + mean^2 - 1 - log(std^2) )
    KLD = 0.5 * torch.sum(std_squared + mean.pow(2) - 1 - 2 * log_std, dim=-1)
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_dimensions = img_shape[1] * img_shape[2] * img_shape[3]  # channels * height * width

    bpd = elbo * torch.log2(torch.exp(torch.tensor(1.0))) / n_dimensions
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    percentiles = torch.linspace(0.5 / grid_size, (grid_size - 0.5) / grid_size, grid_size)

    # Approximate the inverse CDF (icdf) for a standard normal distribution
    z = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * percentiles - 1)

    # Create a 2D grid of latent values
    z1, z2 = torch.meshgrid(z, z, indexing="ij")
    latent_grid = torch.cat([z1.unsqueeze(-1), z2.unsqueeze(-1)], dim=-1).view(-1, 2)

    # Decode each point in the latent space grid
    logits = decoder(latent_grid)  # Decoder outputs logits
    logits = logits.view(-1, 16, 28, 28)  # Reshape to [B, channels, height, width]

    pixel_values = torch.arange(16, device=logits.device).view(1, -1, 1, 1)  # Values: [0, 1, ..., 15]
    expected_intensity = torch.sum(torch.softmax(logits, dim=1) * pixel_values, dim=1)

    # Reshape decoded images to (grid_size**2, channels, height, width)
    decoded_images = expected_intensity.view(grid_size**2, 1, 28, 28)

    # Combine into a single grid of images
    img_grid = torchvision.utils.make_grid(decoded_images, nrow=grid_size, normalize=True, pad_value=1)    
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid

