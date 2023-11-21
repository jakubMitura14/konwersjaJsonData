from torch import nn, Tensor
import torch
import torchvision
import numpy as np
import os
import nnunetv2
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss


class FocalLossV2_orig(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossV2_orig, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss



# from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_focalLoss import FocalLoss
class FocalLossV2(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target,weight):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        # print(f"logit {logit.shape} target {target.shape}")
        
        target_b=torch.clone(target)
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))                     
            
            target_b = target_b.view(target_b.size(0), target_b.size(1), -1)
            target_b = target_b.permute(0, 2, 1).contiguous()
            target_b = target_b.view(-1, target_b.size(-1))
            
            
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)
        if weight.device != logit.device:
            weight = weight.to(logit.device)


        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
            
        # pt = ((one_hot_key * logit)).sum(1) + self.smooth 
        # print(f"one_hot_key {one_hot_key.shape} logit {logit.shape}") #one_hot_key torch.Size([44688, 3]) logit torch.Size([44688, 3] )
        # print(f" one_hot_key[:,0] {one_hot_key[:,0].shape} logit[:,0] {logit[:,0].shape} weight[0] {weight[0].shape}")
        # one_hot_key[:,0]*logit[:,0]*weight[0]
        mask_b= (torch.abs((target_b!=1).float()-0.000000000001))
        # print(f"mask_b {mask_b.shape} one_hot_key[:,0] {one_hot_key[:,0].shape} logit[:,0] {logit[:,0]}")
        # logit= logit*mask_b
        pt= torch.stack([one_hot_key[:,0]*logit[:,0]*weight[0] *mask_b[:,0]
                         , one_hot_key[:,1]*logit[:,2]*weight[1] 
                         ,one_hot_key[:,2]*logit[:,2]*weight[2]*mask_b[:,0]
                         ], dim=1)
        pt = (pt).sum(1) + self.smooth

        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    


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


class Picai_FL_and_CE_loss(nn.Module):
    def __init__(self, fl_kwargs=None, ce_kwargs=None, alpha=0.5, aggregate="sum"):
        super(Picai_FL_and_CE_loss, self).__init__()
        if fl_kwargs is None:
            fl_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}

        self.aggregate = aggregate
        self.fl = FocalLossV2(apply_nonlin=nn.Softmax(), **fl_kwargs)
        # self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = alpha

        self.w0=float(os.getenv('w0'))
        self.w1=float(os.getenv('w1'))
        self.w2=float(os.getenv('w2'))
        self.w_max= np.max(np.array([self.w0,self.w1,self.w2]))


    def forward(self, net_output, target):
        weight=torch.tensor([self.w0,self.w1,self.w2])
        # ce_loss = self.ce(net_output, target)
        fl_loss = self.fl(net_output, target,weight)
        return fl_loss

def _get_deep_supervision_scales(configuration_manager):
    """
    build deep supervision scales
    """
    deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
        configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
    return deep_supervision_scales


def _build_loss(label_manager,is_ddp,configuration_manager):
    """
    copied from main nnunet code
    """
    if label_manager.has_regions:
        loss = DC_and_BCE_loss({},
                                {'batch_dice': configuration_manager.batch_dice,
                                'do_bg': True, 'smooth': 1e-5, 'ddp': is_ddp},
                                use_ignore_label=label_manager.ignore_label is not None,
                                dice_class=MemoryEfficientSoftDiceLoss)
    else:
        loss = DC_and_CE_loss({'batch_dice': configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

    deep_supervision_scales = _get_deep_supervision_scales()
    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss
    weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    weights[-1] = 0

    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    weights = weights / weights.sum()
    # now wrap the loss
    loss = DeepSupervisionWrapper(loss, weights)
    return loss


def build_loss_function(is_lesion_segm
                        ,is_deep_supervision
                        ,is_anatomy_segm
                        ,configuration_manager
                        ,label_manager
                        ,is_priming_segm
                        ):
        """
        builds loss functions diffrent for anatomy and for lesion segmentation and for priming
        """
        is_ddp=False

        if(is_lesion_segm):
            if(is_priming_segm):
                loss= FocalLossV2_orig()
            else:
                loss=Picai_FL_and_CE_loss()


            if(is_deep_supervision):
                deep_supervision_scales = _get_deep_supervision_scales()

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = DeepSupervisionWrapper(loss, weights)

        elif(is_deep_supervision):
            loss = _build_loss()
        elif(is_anatomy_segm):
            loss=DC_and_BCE_loss({},
                                   {'batch_dice': configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': is_ddp},
                                   use_ignore_label=label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
