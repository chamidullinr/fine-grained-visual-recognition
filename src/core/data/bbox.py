import math
from typing import NamedTuple

import numpy as np


__all__ = ['BBox']


def _update_axis(xymin, xymax, diff, max_val):
    new_xymin = xymin - math.floor(diff / 2)
    new_xymax = xymax + math.ceil(diff / 2)
    if new_xymin < 0 or new_xymax > max_val:
        new_xymax -= new_xymin
        new_xymin = 0
    return new_xymin, new_xymax


class BBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def area(self):
        return self.width * self.height

    def copy(self):
        return BBox(self.xmin, self.ymin, self.xmax, self.ymax)

    def numpy(self):
        return np.array([[self.xmin, self.ymin], [self.xmax, self.ymax]])

    def normalize(self, img_height, img_width):
        return BBox(self.xmin / img_width,
                    self.ymin / img_height,
                    self.xmax / img_width,
                    self.ymax / img_height)

    def denormalize(self, img_height, img_width):
        return BBox(int(self.xmin * img_width),
                    int(self.ymin * img_height),
                    int(self.xmax * img_width),
                    int(self.ymax * img_height))
    
    def make_square(self, max_height, max_width):
        height = self.height
        width = self.width
        if height > width:  # change x
            diff = min(height, max_width) - width
            new_xmin, new_xmax = _update_axis(self.xmin, self.xmax, diff, max_width)
            new_bbox = BBox(new_xmin, self.ymin, new_xmax, self.ymax)
        elif width > height:  # change y
            diff = min(width, max_height) - height
            new_ymin, new_ymax = _update_axis(self.ymin, self.ymax, diff, max_height)
            new_bbox = BBox(self.xmin, new_ymin, self.xmax, new_ymax)
        else:
            new_bbox = self

        return new_bbox

    def make_min_size(self, min_height, min_width, max_height, max_width):
        height = self.height
        width = self.width
        if min_width > width:
            diff = min_width - width
            new_xmin, new_xmax = _update_axis(self.xmin, self.xmax, diff, max_width)
        else:
            new_xmin, new_xmax = self.xmin, self.xmax

        if min_height > height:
            diff = min_height - height
            new_ymin, new_ymax = _update_axis(self.ymin, self.ymax, diff, max_height)
        else:
            new_ymin, new_ymax = self.ymin, self.ymax

        return BBox(new_xmin, new_ymin, new_xmax, new_ymax)
