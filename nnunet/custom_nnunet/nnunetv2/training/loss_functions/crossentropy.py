from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.Module):
    """
    we will ignore the 2 - put weight nearly 0 on it and take as binary cross entrophy 1
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        target_a= (target==1)
        weight= (target!=2)
        return nn.functional.binary_cross_entropy(input, target_a, weight=weight)


