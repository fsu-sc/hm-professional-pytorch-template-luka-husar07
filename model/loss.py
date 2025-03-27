import torch
import torch.nn as nn


def mse_loss(output, target):
    return nn.MSELoss()(output, target)

def nll_loss(output, target):
    return nn.NLLLoss()(output, target)
