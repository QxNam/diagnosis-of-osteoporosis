import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        p = torch.exp(logp)
        ce_loss = F.nll_loss(logp, targets, reduction='none')
        focal_term = (1 - p.gather(1, targets.unsqueeze(1)).squeeze()) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_term *= alpha_t
        loss = focal_term * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        true_dist = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        true_dist = true_dist * (1 - self.smoothing) + self.smoothing / n_classes
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

def get_loss_function(name, config):
    if name == "focal":
        alpha = config.get("focal_alpha", None)
        if alpha is not None:
            alpha = torch.tensor(alpha).to(config['device'])
        return FocalLoss(gamma=config.get("focal_gamma", 2.0), alpha=alpha)
    elif name == "labelsmoothing":
        return LabelSmoothingCrossEntropy(smoothing=config.get("smoothing", 0.1))
    else:
        return nn.CrossEntropyLoss()
