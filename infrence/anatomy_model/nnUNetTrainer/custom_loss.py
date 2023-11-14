from torch import nn, Tensor
import torch
import torchvision
import numpy as np
import os


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


class RobustCrossEntropyLosss(nn.Module):
    """
    we will ignore what is 1 but not 2 - put weight nearly 0 on it and take as binary cross entrophy 1
    """
    def __init__(self):
        super().__init__()
        # self.main_focal= FocalLossV2(gamma=0.7)

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
        self.fl = FocalLossV2(apply_nonlin=nn.Softmax(), **fl_kwargs)
        # self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = alpha

        self.w0=float(os.getenv('w0'))
        self.w1=float(os.getenv('w1'))
        self.w2=float(os.getenv('w2'))
        self.w_max= np.max(np.array([self.w0,self.w1,self.w2]))


    def forward(self, net_output, target):
        # print(f"nnnnnn net_output {net_output.shape} target {target.shape} ") #nnnnnn net_output torch.Size([19, 3, 28, 48, 56]) target torch.Size([19, 1, 28, 48, 56]) 
        # net_output=net_output[:,0:2,:,:,:]#torch.stack([net_output[:,0,:,:,:],net_output[:,2,:,:,:] ] , dim=1) 
        # target_mask=torch.abs(((target!=1).float()-0.00000001 ))
        #         # target_mask=(((target==0).float()*5)+((target==1).float()*0.1)+((target==2).float()*10))/10

        # net_output=net_output*target_mask
        # target=(target==2).int()
        weight=torch.tensor([self.w0,self.w1,self.w2])
        # ce_loss = self.ce(net_output, target)
        fl_loss = self.fl(net_output, target,weight)
        return fl_loss
        # if self.aggregate == "sum":
        #     result = self.alpha*fl_loss + (1-self.alpha)*ce_loss
        # else:
        #     raise NotImplementedError("nah son")
        # return result