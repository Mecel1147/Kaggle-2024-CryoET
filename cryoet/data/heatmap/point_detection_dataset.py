import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cryoet.data.functional import normalize_volume_to_unit_range
from cryoet.data.parsers import (
    get_volume_and_objects,
    ANGSTROMS_IN_PIXEL,
)


class CryoETPointDetectionDataset(Dataset):
    def __init__(self, root, study, mode, split="train"):
        volume_data, object_centers, object_labels, object_radii = get_volume_and_objects(
            root_dir=root,
            study_name=study,
            mode=mode,
            split=split,
        )

        self.study = study
        self.split = split
        self.mode = mode
        self.volume_data = normalize_volume_to_unit_range(volume_data)
        self.volume_shape = volume_data.shape
        self.object_centers = object_centers
        self.object_labels = object_labels
        self.object_radii = object_radii

        self.object_centers_px = object_centers / ANGSTROMS_IN_PIXEL
        self.object_radii_px = object_radii / ANGSTROMS_IN_PIXEL


# 该函数用于生成三维高斯热力图，用于在heatmap中标记目标中心点位置
# 在深度学习任务中，这些热力图可以作为监督信号，指导模型学习目标的空间分布特征。
# 输出一个(D, H, W)的三维高斯热力图，中间为1，周围逐渐减小到0
def centernet_gaussian_3d(shape, sigma=1.0):
    # shape is [D, H, W]
    # sigma是标准差，越大越平滑
    # 在奇数尺寸的核中，准确地对称地构造中心点到边界的距离
    d, m, n = [(ss - 1.0) / 2.0 for ss in shape]
    # 生成三维坐标网格，z.shape = (11, 1, 1)
	# y.shape = (1, 11, 1)
	# x.shape = (1, 1, 11)
    # 得到的 z, y, x 组合后可广播为 (11, 11, 11) 的网格张量，每个位置表示相对中心的 (dz, dy, dx) 偏移。
    z, y, x = np.ogrid[-d : d + 1, -m : m + 1, -n : n + 1]

    # 计算高斯函数的值
    # 公式为 exp(-(z^2 + y^2 + x^2) / (2 * sigma^2))
    h = np.exp(-(z * z + x * x + y * y) / (2 * sigma * sigma))
    # 把极小的值置为0
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # Place 1.0 in the center of the gaussian (just in case)
    # 强制将中心点的值设为1.0，防止数值精度问题导致中心点的值略小于1
    h[h.shape[0] // 2, h.shape[1] // 2, h.shape[2] // 2] = 1.0

    return h


# 把多个目标点画到一个heatmap中，每个点用一个3d高斯球表示，按类别写入不同的通道
# 用于生成训练标签
def encode_centers_to_heatmap(centers, labels, radii, shape, num_classes):
    # 所有值初始化为0，每个类别对应一个通道
    heatmap = np.zeros((num_classes,) + shape, dtype=np.float32)

    # 将中心点和半径四舍五入
    depth, height, width = shape
    centers = (centers + 0.5).astype(int)
    radii = (radii + 0.5).astype(int)

    for center, label, radius in zip(centers, labels, radii):
        x, y, z = center
        # z, y, x = int(z + 0.5), int(y + 0.5), int(x + 0.5)
        # radius = int(radius + 0.5)

        # 生成高斯图
        diameter = 2 * radius + 1
        gaussian = centernet_gaussian_3d((diameter, diameter, diameter), sigma=diameter / 3.0)

        # 对高斯核裁剪
        # radius +1 表示从中心（包括中心）要往后贴radius + 1 个voxel，也就是需要多少个
        # depth - z 表示从中心点到末尾的剩余空间，也就是能贴多少个voxel
        front, back = min(z, radius), min(depth - z, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        left, right = min(x, radius), min(width - x, radius + 1)

        # 把高斯核贴进heatmap
        # 当前种类热力图  
        masked_heatmap = heatmap[label, z - front : z + back, y - top : y + bottom, x - left : x + right]
        # 生成的高斯核裁出的热力图
        masked_gaussian = gaussian[
            radius - front : radius + back,
            radius - top : radius + bottom,
            radius - left : radius + right,
        ]
        # if用于安全检查，
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            # 取最大值
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


# 从模型输出的heatmap中找到所有可能的目标中心点，包括位置、类别和分数
def decoder_centers_from_heatmap(probas: Tensor, kernel=3, top_k=256):
    """

    :param probas: [B,C,D,H,W] (after sigmoid)
    :param kernel:
    :param top_k: N - number of top candidates per class
    :return:
        Scores [B, N]
        Labels [B, N]
        Coords [B, N, 3] - Coordinates of each peak
    """

    # nms
    pad = (kernel - 1) // 2
    maxpool = torch.nn.functional.max_pool3d(probas, kernel_size=kernel, padding=pad, stride=1)

    mask = probas == maxpool

    peaks = probas * mask

    batch, cat, depth, height, width = peaks.size()

    # 选出topk目标点
    topk_scores, topk_inds = torch.topk(peaks.view(batch, cat, -1), top_k)

    # 记录类别
    topk_clses = torch.arange(cat, device=probas.device).view(1, -1, 1)
    topk_clses = topk_clses.expand(batch, -1, top_k)

    # 把faltten的索引还原成z，y,x坐标
    topk_inds = topk_inds % (depth * height * width)
    topk_zs = topk_inds // (width * height)
    topk_inds = topk_inds % (width * height)
    topk_ys = topk_inds // width
    topk_xs = topk_inds % width

    # 构造输出结果
    # Gather scores for a specific class
    # B, C, N -> B, N
    topk_scores = topk_scores.reshape(batch, -1)
    # 与scores对应，记录类别
    topk_clses = topk_clses.reshape(batch, -1)
    topk_ys = topk_ys.view(batch, -1)
    topk_xs = topk_xs.view(batch, -1)
    topk_zs = topk_zs.view(batch, -1)

    return topk_scores, topk_clses, torch.stack([topk_xs, topk_ys, topk_zs], dim=-1)
