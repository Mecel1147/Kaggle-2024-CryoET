import dataclasses
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclasses.dataclass
class PointDetectionOutput:
    logits: Tensor
    loss: Optional[Tensor]


def quality_focal_loss(logits, labels, num_items_in_batch=None, alpha=4, **kwargs):
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    qfl_term = (logits.sigmoid() - labels).abs().pow(alpha)
    loss = bce_loss * qfl_term

    loss = loss.sum()
    if num_items_in_batch is None:
        num_items_in_batch = labels.eq(1).sum().item()

    num_items_in_batch = max(num_items_in_batch, 1)
    return loss.div_(num_items_in_batch)


# 训练heatmap的默认损失函数
def point_detection_loss(logits, labels, num_items_in_batch=None, alpha=2, beta=4, eps=1e-6, **kwargs):
    """
    Compute centernet loss for object detection

    :param logits: Predicted headmap logits BxCxDxHxW
    :param labels: Target heatmap BxCxDxHxW
    :return: Single scalar loss
    """
    # 计算mask
    pos_mask = labels.eq(1)
    neg_mask = ~pos_mask

    # 把logits转换成概率，clamp用于限制logits的范围，避免log(0)的情况
    pt = logits.sigmoid().clamp(eps, 1 - eps)

    # pos_loss = - (1 - pt)^α * log(sigmoid(logits)) * pos_mask
    # alpha=2.0，表示对难点更关注
    pos_loss: Tensor = -torch.pow(1 - pt, alpha) * F.logsigmoid(logits) * pos_mask
    # neg_loss = - (1 - label)^β * pt^α * log(1 - sigmoid(logits)) * neg_mask
    # 让模型在背景区域保持低输出值
    neg_loss: Tensor = -torch.pow(1.0 - labels, beta) * torch.pow(pt, alpha) * F.logsigmoid(-logits) * neg_mask

    loss = (neg_loss + pos_loss).sum()
    
    # 防止➗0
    if num_items_in_batch is None:
        num_items_in_batch = pos_mask.sum().item()

    num_items_in_batch = max(num_items_in_batch, 1)
    return loss.div_(num_items_in_batch)


class PointDetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, use_qfl_loss: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1)
        self.use_qfl_loss = use_qfl_loss
        torch.nn.init.zeros_(self.conv.weight)
        torch.nn.init.constant_(self.conv.bias, -3)

    def forward(self, features, labels=None, **loss_kwargs):
        logits = self.conv(features)

        if torch.jit.is_tracing():
            return logits

        loss = None
        if labels is not None:
            if self.use_qfl_loss:
                loss = quality_focal_loss(logits.float(), labels.float(), **loss_kwargs)
            else:
                loss = point_detection_loss(logits.float(), labels.float(), **loss_kwargs)

        return PointDetectionOutput(logits=logits, loss=loss)
