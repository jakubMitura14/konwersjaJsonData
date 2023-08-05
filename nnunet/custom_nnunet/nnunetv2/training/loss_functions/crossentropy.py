from torch import nn, Tensor
import torch

class RobustCrossEntropyLoss(nn.Module):
    """
    we will ignore what is 1 but not 2 - put weight nearly 0 on it and take as binary cross entrophy 1
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        target_a= (target==2)
        weight= torch.abs(((~(target==1)).float()-0.000001))
        return nn.functional.binary_cross_entropy(input, target_a, weight=weight)


