import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from collections import OrderedDict
from myrllib.policies.policy import Policy, weight_init

class CategoricalMLPPolicy(Policy):
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu):
        super(CategoricalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.logits = nn.Linear(layer_sizes[-1], output_size)
        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                    bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        logits = F.linear(output, weight=params['logits.weight'], bias=params['logits.bias'])

        return Categorical(logits=logits)

