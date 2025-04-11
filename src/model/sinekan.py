import torch
import torch.nn.functional as F
import math
from typing import *

class SineKANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', grid_size=8, is_first=False, add_bias=True, norm_freq=True):
        super(SineKANLayer,self).__init__()
        self.grid_size = grid_size
        self.device = device
        self.is_first = is_first
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.grid_norm_factor = (torch.arange(grid_size) + 1)
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)
            
        if is_first:
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim  / self.grid_norm_factor)
        else:
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim  / self.grid_norm_factor)

        grid_phase = torch.linspace(0, math.pi, grid_size).reshape(1, 1, 1, grid_size)
        self.input_phase = torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)

        if norm_freq:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        grid_phase_perturbation = torch.rand((1, 1, 1, self.grid_size)) / (self.grid_size) * 0.1
        input_phase_perturbation = torch.rand((1, 1, self.input_dim, 1)) / (self.input_dim) * 0.1
        
        phase = grid_phase.to(device) + self.input_phase + grid_phase_perturbation.to(device) + input_phase_perturbation.to(device)

        self.register_buffer('phase', phase)
        
        if self.add_bias:
            self.bias  = torch.nn.Parameter(torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y

class SineKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8,
        device: str = 'cuda',
    ) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList([
            SineKANLayer(
                in_dim, out_dim, device, grid_size=grid_size, is_first=True
            ) if i == 0 else SineKANLayer(
                in_dim, out_dim, device, grid_size=grid_size,
            ) for i, (in_dim, out_dim) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:]))
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x