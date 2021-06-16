"""Custom loss for long tail problem.

- Author: Junghoon Kim
- Email: placidus36@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.torch_utils import CustomLoss,LabelSmoothingLoss


class CustomCriterion:
    """Custom Criterion."""

    def __init__(self, samples_per_cls, device, fp16=True, loss_type="softmax"):
        if not samples_per_cls:
            loss_type = "softmax"
        else:
            self.samples_per_cls = samples_per_cls
            self.frequency_per_cls = samples_per_cls / np.sum(samples_per_cls)
            round_list = []
            round_list_two = []
            for i in self.frequency_per_cls:
                round_list.append(round(i,3))
            for i in 1/self.frequency_per_cls:
                round_list_two.append(round(i,1))
                
            self.frequency_per_cls = np.array(round_list)
            self.reverse_weight = torch.tensor(round_list_two)
            
            self.no_of_classes = len(samples_per_cls)
        self.device = device
        self.fp16 = fp16

        if loss_type == "softmax":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == "logit_adjustment_loss":
            tau = 1.0
            self.logit_adj_val = (
                torch.tensor(tau * np.log(self.frequency_per_cls))
                .float()
                .to(self.device)
            )
            self.logit_adj_val = (
                self.logit_adj_val.half() if fp16 else self.logit_adj_val.float()
            )
            self.logit_adj_val = self.logit_adj_val.to(device)
            
            self.criterion = self.logit_adjustment_loss
        elif loss_type == "customloss":
            self.criterion = CustomLoss()
        elif loss_type == "weighted":
            self.reverse_weight = self.reverse_weight.to(device)
            self.reverse_weight = (
                self.reverse_weight.half() if fp16 else self.reverse_weight.float()
            )
            self.reverse_weight = self.reverse_weight.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=self.reverse_weight)
            
        elif loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingLoss()

    def __call__(self, logits, labels):
        """Call criterion."""
        return self.criterion(logits, labels)

    def logit_adjustment_loss(self, logits, labels):
        """Logit adjustment loss."""
             
        logits_adjusted = logits + self.logit_adj_val.repeat(labels.shape[0], 1)
        loss = F.cross_entropy(input=logits_adjusted, target=labels)
            
        return loss