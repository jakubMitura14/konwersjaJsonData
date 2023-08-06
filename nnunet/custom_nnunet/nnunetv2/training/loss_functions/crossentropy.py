from torch import nn, Tensor
import torch
import torchvision

from focal_loss.focal_loss import FocalLoss



class RobustCrossEntropyLosss(nn.Module):
    """
    we will ignore what is 1 but not 2 - put weight nearly 0 on it and take as binary cross entrophy 1
    """
    def __init__(self):
        super().__init__()
        self.main_focal= FocalLoss(gamma=0.7)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(f"input {input.shape} target {target.shape} target {target.min()} {target.max()}")
        target_a= (target==2.0)
        target_a=target_a[:,0,:,:,:].float()
        input=input[:,2,:,:,:]
        weight= (torch.abs((((target!=1.0)[:,0,:,:,:] ).float()-0.00000001))+ target_a*20)/21 # TODO() try increasing weight for the center

        # return  self.main_focal( (torch.nn.functional.sigmoid(input)*weight),target_a)
        focal_loss= torch.mean(torchvision.ops.sigmoid_focal_loss((input*weight),target_a))
        cross_loss=  nn.functional.binary_cross_entropy_with_logits(input, target_a, weight=weight)
        return cross_loss+focal_loss

