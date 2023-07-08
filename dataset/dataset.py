# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import os

import numpy as np
from numpy.core.records import record
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from dataset.transform import __get_transforms
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

def get_dataloader(c_data, batch_size=1):
    # 图片处理
    tfs_spatial, tfs_label = __get_transforms(c_data)

    read_label_func = None
    if c_data.name == "activitynet1.3":
        read_label_func = read_label_activitynet
    elif c_data.name == "minikinetics":
        read_label_func = read_label_minikinetics
    elif c_data.name == "ucf101":
        read_label_func = read_label_ucf101
    elif c_data.name == "hmdb51":
        read_label_func = read_label_hmdb51
    elif c_data.name == "ssv2":
        read_label_func = read_label_ssv2

    dataset = VideoLoader(
        c_data.path_split,
        c_data.path_frame,
        c_data.path_label,
        c_data.path_classid,
        read_label_func=read_label_func,
        transform_spatial=tfs_spatial,
        transform_label=tfs_label,
        clip_length=c_data.num_frames,
    )

    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=c_data.shuffle,
        num_workers=c_data.num_workers,
        drop_last=True
    )

    return dataloader


class VideoLoader(data.Dataset):
    def __init__(
        self,
        path_split,
        path_frames,
        path_label,
        path_classid,
        read_label_func=None,
        clip_length=10,
        transform_spatial=None,
        transform_label=None,
    ):
        super().__init__()
        self.path_split = path_split
        self.path_frames = path_frames
        self.path_label = path_label
        self.path_classid = path_classid

        self.transform_spatial = transform_spatial
        self.transform_label = transform_label
        self.clip_length = clip_length

        # 验证集视频
        if read_label_func == read_label_ucf101:
            self.root_path = r'../UCF101/jpg/'
            self.image_tmpl = 'img_{:05d}.jpg'
            video_keys = [line.strip().split(' ')[0] for line in open(path_split, "r")]
        elif read_label_func == read_label_hmdb51:
            self.root_path = r'../HMDB51/jpg/'
            self.image_tmpl = 'img_{:05d}.jpg'
            video_keys = [line.strip().split(' ')[0] for line in open(path_split, "r")]

        elif read_label_func == read_label_ssv2:
            self.root_path = r'../ssv2/ssv2-frames/'
            self.image_tmpl = '{:06d}.jpg'
            video_keys = [line.strip().split(' ')[0] for line in open(path_split, "r")]
        else:
            video_keys = [line.strip() for line in open(path_split, "r")]
        # (验证集)视频个数
        self.number_of_videos = len(video_keys)

        # label
        # self.video_context: return metedata
        # metadata = {"video_name":{"label":标签, "cc":是否有共享许可}}
        self.video_context = read_label_func(self.path_label)
        # 类和索引建立为字典
        # class_id = {line.strip(): i for i, line in enumerate(open(path_classid, "r"))}
        # (vname, context) = (key, value)
        # 这一段在给验证集视频打标签:将字符串标签转化为索引
        # for vname, context in self.video_context.items():
        #     if vname in video_keys:
        #         self.video_context[vname]["label"] = [
        #             class_id[line] for line in context["label"]
        #         ]
        
        # frame
        # video_info = {"视频名称"：[每一帧的名称, ...]}
        # video_info = {}
        # for vname in video_keys:
        #     if os.path.isdir(os.path.join(path_frames, vname)):
        #         frame_names = glob.glob(os.path.join(path_frames, vname) + "/*.jpg")
        #         if len(frame_names) > 0:
        #             # 当前视频文件夹下的所有帧名称以列表形式存储
        #             video_info[vname] = frame_names
        # print(video_keys[0])
        # # 从这些帧内均匀采样
        # # self.clip_indices存储每个视频采样出来的帧[[视频1的帧],[视频2的帧], ...], 
        # # self.clip_names存储对应视频的名称[[视频1名称], [视频2名称], ...]
        # self.clip_indices, self.clip_names = sample_frames_uniform(
        #     video_info, clip_length=self.clip_length
        # )
        # print(self.clip_indices, self.clip_names)
        self._parse_list()

    def __getitem_label__(self, frame_list, vname):
        metadata = {"frame_ids": [], "labels": []}
        for frame_name in frame_list:
            metadata["labels"].append(self.video_context[vname])
            metadata["frame_ids"].append(frame_name)

        labels = metadata["labels"]
        if self.transform_label is not None:
            labels = self.transform_label(labels)

        labels = torch.stack(labels)

        return labels, metadata

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.path_split)]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = sample_frames_uniform_(record, self.clip_length) 
        return self.get(record, segment_indices)

    def _load_image(self, directory, idx):
        try:
            # .convert("RGB")转化为RGB三通道
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]   

    def get(self, record, indices):
        images = list()
        for seg_ind in indices.tolist():
            p = int(seg_ind)
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)

        process_data = self.transform_spatial(images)
        process_data = torch.stack(process_data)
        return process_data, record.label
    
    def __len__(self):
        return len(self.video_list)

def sample_frames_uniform_(record, clip_length):
    
    video_length = record.num_frames
    clip_length = clip_length
    indices = np.clip(
        np.linspace(1, video_length + 1, clip_length), 0, video_length - 1
    ).astype("int")

    return indices

def sample_frames_uniform(video_indices, clip_length):
    """
    selects one clip of length clip_length uniformly from a video
    从视频中均匀选择一个长度为clip_length的一段
    """
    splits, split_names = [], []
    # 对每个视频均匀采样一些帧放到splits, 对应的视频名称存在split_names
    for video_name, frame_list in sorted(video_indices.items()):
        video_frames = np.array(sorted(frame_list))
        video_length = len(video_frames)
        # np.clip():param1 表示要处理的列表，小于param2的都变为param2,大于param3的都变为param3
        # 步长为clip_length(此处为10), 每10帧取一帧，假如0到100帧，取0, 11, 22, 33, 44, ..., 99(np.clip处理防止越界)
        indices = np.clip(
            np.linspace(0, video_length, clip_length), 0, video_length - 1
        ).astype("int")
        # 把video_frame里面的所有indices索引所指的帧名称都取出来作为列表
        # video_frames 和 indices都是array类型
        split = video_frames[indices].tolist()
        # 每一段加入splits
        splits.append(split)
        split_names.append(video_name)

    return splits, split_names


def read_label_activitynet(path_labelfile):
    # read label file
    with open(path_labelfile) as json_file:
        dataset = json.load(json_file)
    dataset = dataset["database"]

    metadata = {}
    for key, val in dataset.items():
        vkey = "v_%s.mp4" % key
        labels = [ann["label"] for ann in dataset[key]["annotations"]]
        segments = [ann["segment"] for ann in dataset[key]["annotations"]]
        cur_cntx = {"label": labels, "segment": segments, "set": dataset[key]["subset"]}
        metadata[vkey] = cur_cntx

    return metadata

def read_label_minikinetics(path_labelfile):
    # [1:] to skip csv header
    # metadata = {"video_name":{"label":标签, "cc":是否有共享许可}}
    data = [x.strip() for x in open(path_labelfile, "r")][1:]
    
    metadata = {}
    for i, line in enumerate(data):
        label, ytname, b, e, split, cc = line.strip().split(",")
        if label[0] == '"':
            label = label[1:-1]
        vkey = "%s_%s_%s.mp4" % (
            ytname,
            b.split(".")[0].zfill(6),
            e.split(".")[0].zfill(6),
        )
        cur_cntx = {"label": [label], "cc": cc}
        metadata[vkey] = cur_cntx

    return metadata


# add for UCF101 & HMDB51 dataset
def read_label_ucf101(path_labelfile):
    data = [x.strip() for x in open(path_labelfile, "r")]
    metadata = {}
    for _, line in enumerate(data):
        video_name, frame_nums, class_id = line.strip().split(" ")
        vkey = video_name
        cur_cntx = {"label": int(class_id)}
        metadata[vkey] = cur_cntx
    return metadata

def read_label_hmdb51(path_labelfile):
    data = [x.strip() for x in open(path_labelfile, "r")]
    metadata = {}
    for _, line in enumerate(data):
        video_name, frame_nums, class_id = line.strip().split(" ")
        vkey = video_name
        cur_cntx = {"label": int(class_id)}
        metadata[vkey] = cur_cntx
    return metadata
# add for something-something v2 dataset
def read_label_ssv2(path_labelfile):
    data = [x.strip() for x in open(path_labelfile, "r")]
    metadata = {}
    for _, line in enumerate(data):
        video_name, class_id, frame_nums = line.strip().split(" ")
        vkey = video_name
        cur_cntx = {"label": int(class_id)}
        metadata[vkey] = cur_cntx
    return metadata