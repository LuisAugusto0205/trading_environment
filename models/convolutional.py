import numpy as np

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override

from ray.rllib.models.torch.misc import SlimConv2d, SlimFC

from torch import nn
torch, _ = try_import_torch()

default_conv_model_config = {
    "custom_model": "ConvTorchModel",
    "custom_model_config": {
        "channels": [5, 50],
        "kernels": [(3, 1), (13, 1)],
        "activation_fn": ["relu", "relu"]
    },
    "no_final_linear": True
}

class ConvModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.num_outputs = num_outputs
        self._last_batch_size = None

        layers = []
        last_channel = 1
        n_channels = model_config['custom_model_config']['channels']
        kernels = model_config['custom_model_config']['kernels']
        act_funcs = model_config['custom_model_config']['activation_fn']

        for n_channel, kernel, act_func in zip(n_channels, kernels, act_funcs):
            layers.append(SlimConv2d(
                in_channels = last_channel,
                out_channels = n_channel,
                kernel = kernel,
                stride = 1,
                activation_fn=act_func,
                padding = 0
            ))
            last_channel = n_channel
        
        self._convs = nn.Sequential(*layers)
        self._fc = SlimFC(last_channel + num_outputs, num_outputs)

        layers = []
        last_channel = 1
        for n_channel, kernel, act_func in zip(n_channels, kernels, act_funcs):
            layers.append(SlimConv2d(
                in_channels = last_channel,
                out_channels = n_channel,
                kernel = kernel,
                stride = 1,
                activation_fn=act_func,
                padding = 0
            ))
            last_channel = n_channel
        
        self._convs_value = nn.Sequential(*layers)

        self._fc_value = SlimFC(last_channel + num_outputs, 1)
        
    #@override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)

        conv_input = self._last_flat_in[:, :-self.num_outputs].unsqueeze(dim=1).unsqueeze(dim=-1)
        conv_output = self._convs(conv_input)

        # with open('debung.txt', 'a') as file:
        #     file.write(f"{conv_output.shape}\n")
        action_output = self._fc(torch.cat([conv_output.squeeze(dim=[2, 3]), self._last_flat_in[:, -self.num_outputs:]], dim=1))

        return action_output, state

    #@override(ModelV2)
    def value_function(self):
        conv_input = self._last_flat_in[:, :-self.num_outputs].unsqueeze(dim=1).unsqueeze(dim=-1)
        conv_output = self._convs_value(conv_input)
        value_output = self._fc_value(torch.cat([conv_output.squeeze(dim=[2, 3]), self._last_flat_in[:, -self.num_outputs:]], dim=1))
        
        return value_output.reshape(-1)