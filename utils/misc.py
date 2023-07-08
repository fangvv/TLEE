# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


class map_output_transform:
    def __init__(self, video_level=False):
        self.video_level = video_level

    def __call__(self, output):
        probs, y_mh, ids = output
        if self.video_level:
            # rsplit() 类似split, 但是从字符串后面开始分割
            ids = [x.rsplit("_", 1)[0] for x in ids]
        return probs.squeeze(dim=1), y_mh, ids
