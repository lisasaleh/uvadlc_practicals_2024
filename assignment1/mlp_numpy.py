################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.layers = []

        # Add hidden layers
        in_features = n_inputs
        for out_features in n_hidden:
            
            # Add Linear and ELU modules for each hidden layer
            self.layers.append(LinearModule(in_features, out_features))
            self.layers.append(ELUModule(alpha=1.0))  #  alpha=1.0 for ELU
            in_features = out_features

        # Add the output layer
        self.layers.append(LinearModule(in_features, n_classes))
        self.layers.append(SoftMaxModule())  # Add softmax for classification
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Flatten the input to shape (batch_size, n_inputs)
        x = x.reshape(x.shape[0], -1)  # Flatten each sample into a 1D vector (3072,)
        
        # Pass input through all layers
        for layer in self.layers:
            x = layer.forward(x)
        
        out = x  # Final output
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Pass gradients backward through each layer
        for layer in reversed(self.layers):
          dout = layer.backward(dout)

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Call clear_cache for each layer
        for layer in self.layers:
          layer.clear_cache()          
        #######################
        # END OF YOUR CODE    #
        #######################
