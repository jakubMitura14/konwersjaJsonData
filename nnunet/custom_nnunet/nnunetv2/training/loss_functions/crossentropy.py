from torch import nn, Tensor
import torch
import torchvision

from focal_loss.focal_loss import FocalLoss
from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_focalLoss import FocalLoss

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


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



class Picai_FL_and_CE_loss(nn.Module):
    def __init__(self, fl_kwargs=None, ce_kwargs=None, alpha=0.5, aggregate="sum"):
        super(Picai_FL_and_CE_loss, self).__init__()
        if fl_kwargs is None:
            fl_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}

        self.aggregate = aggregate
        self.fl = FocalLoss(apply_nonlin=nn.Softmax(), **fl_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = alpha

    def forward(self, net_output, target):
        fl_loss = self.fl(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = self.alpha*fl_loss + (1-self.alpha)*ce_loss
        else:
            raise NotImplementedError("nah son")
        return result
