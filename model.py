from torch import Tensor, nn
import torch
from torch.nn import functional as F

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        if 'ViT' in model:
            #self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)
            #self.conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            #self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)
            #self.conv = nn.Conv2d(dim_in * 2 ** (i + 2), dim_out, kernel_size=1)
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                # tokens[i] = tokens[i][:, 1:, :].permute(0, 2, 1).unsqueeze(-1)
                # #tokens[i] = tokens[i][:, 1:, :].unsqueeze(-1)
                # tokens[i] = self.conv(tokens[i])
                # tokens[i] = tokens[i].squeeze(-1).permute(0, 2, 1)
                # #tokens[i] = self.fc[i](tokens[i])

                tokens[i] = self.fc[i](tokens[i][:, 1:, :])

            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.conv(tokens[i])
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
