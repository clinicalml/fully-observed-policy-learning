import torch
import torch.nn as nn
import torch.nn.functional as F


# Convex `cross-entropy' policy loss
def policy_loss(output, target):
    return -1*torch.mean(
        (torch.log(output) * target)
    )

# Non-convex loss: computed expected reward exactly
def policy_loss_non_convex(output, target):
    return -1*torch.mean(output * target)
