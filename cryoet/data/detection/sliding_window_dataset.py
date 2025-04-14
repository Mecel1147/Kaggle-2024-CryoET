import numpy as np

from .detection_dataset import CryoETObjectDetectionDataset
from .mixin import ObjectDetectionMixin
from ..functional import compute_better_tiles_with_num_tiles
from ..parsers import AnnotatedVolume
from ...training.args import DataArguments, ModelArguments


# 吧一个大的3D CryoET图像切成一块块固定大小的“小图像块”，并自动匹配每块中的目标位置和类别信息
class SlidingWindowCryoETObjectDetectionDataset(CryoETObjectDetectionDataset, ObjectDetectionMixin):

    def __init__(
        self,
        sample: AnnotatedVolume,
        data_args: DataArguments,
        model_args: ModelArguments,
    ):
        super().__init__(sample)

        self.window_size = (
            model_args.valid_depth_window_size,
            model_args.valid_spatial_window_size,
            model_args.valid_spatial_window_size,
        )
        self.num_tiles = (
            model_args.valid_depth_num_tiles,
            model_args.valid_spatial_num_tiles,
            model_args.valid_spatial_num_tiles,
        )

        # 把一个大的(Z, Y, X)的3D CryoET图像切成nums_tiles个小窗口
        # 每个tile的大小是window_size
        # 返回一组(z_slice, y_slice, x_slice)
        self.tiles = list(
            compute_better_tiles_with_num_tiles(self.sample.volume_shape, window_size=self.window_size, num_tiles=self.num_tiles)
        )
        self.data_args = data_args

    # 第 i 块 tile 的图像和标签怎么取
    def __getitem__(self, idx):
        tile = self.tiles[idx]  # tiles are z y x order
        centers_px = self.sample.centers_px  # x y z
        radii_px = self.sample.radius_px
        object_labels = self.sample.labels

        # Crop the centers to the tile
        # fmt: off
        centers_x, centers_y, centers_z = centers_px[:, 0], centers_px[:, 1], centers_px[:, 2]
        keep_mask = (
                (centers_z >= tile[0].start) & (centers_z < tile[0].stop) &
                (centers_y >= tile[1].start) & (centers_y < tile[1].stop) &
                (centers_x >= tile[2].start) & (centers_x < tile[2].stop)
        )
        # fmt: on

        volume = self.sample.volume[tile[0], tile[1], tile[2]].copy()
        centers_px = centers_px[keep_mask].copy() - np.array([tile[2].start, tile[1].start, tile[0].start]).reshape(1, 3)
        radii_px = radii_px[keep_mask].copy()
        object_labels = object_labels[keep_mask].copy()

        # Pad the volume to the window size
        pad_z = self.window_size[0] - volume.shape[0]
        pad_y = self.window_size[1] - volume.shape[1]
        pad_x = self.window_size[2] - volume.shape[2]

        volume = np.pad(
            volume,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )

        data = self.convert_to_dict(
            volume=volume,
            centers=centers_px,
            labels=object_labels,
            radii=radii_px,
            tile_offsets_zyx=(tile[0].start, tile[1].start, tile[2].start),
            study_name=self.sample.study,
            mode=self.sample.mode,
            volume_shape=self.sample.volume_shape,
        )

        return data

    # 有多少块 tile
    def __len__(self):
        return len(self.tiles)
