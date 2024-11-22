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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.

        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        # Store layers explicitly
        self.hidden_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList() if use_batch_norm else None
        self.activation_layers = nn.ModuleList()

        # Define input to the first hidden layer
        in_features = n_inputs
        for out_features in n_hidden:
            # Add Linear layer
            linear_layer = nn.Linear(in_features, out_features)
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in')
            self.hidden_layers.append(linear_layer)

            # Add BatchNorm layer
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(out_features))

            # Add ELU activation
            self.activation_layers.append(nn.ELU())

            # Update input size for the next layer
            in_features = out_features

        # Define and init output layer
        self.output_layer = nn.Linear(in_features, n_classes)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in')
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
        x = x.view(x.size(0), -1)  # Flatten all dimensions except the batch size
        
        # Pass input through each hidden layer, applying BatchNorm and ELU if specified
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            if self.batch_norm_layers:     # Apply batch normalization if enabled
                x = self.batch_norm_layers[i](x)
            x = self.activation_layers[i](x)

        # Final output layer
        out = self.output_layer(x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device

