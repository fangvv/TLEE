# TLEE

This is the source code for our paper: **TLEE: Temporal-wise and Layer-wise Early Exiting Network for Efficient Video Recognition on Edge Devices**. A brief introduction of this work is as follows:

> With the explosive growth in video streaming comes a rising demand for efficient and scalable video understanding. State-of-the-art video recognition approaches based on Convolutional Neural Network (CNN) have shown promising performance by adopting 2D or 3D CNN architectures. However, the large data volumes, high resource demands, and strict latency requirements have hindered the wide application of these solutions on resource-constrained Internet-of-Things (IoT) and edge devices. To address this issue, we propose a novel framework called TLEE that enables the input samples the abilities of both Temporal-wise and Layer-wise Early Exiting on 2D CNN backbones for efficient video recognition. TLEE consists of three types of modules: gating module, branch module, and feature reuse module. The gating module determines for an input video from which frame of this video to exit the per-frame computation, while the branch module determines for an input frame from which layer of the CNN backbone to exit the per-layer computation. Besides, based on the accumulated features of frame sequences from exit branches, the feature reuse module generates effective video representations to enable more efficient predictions. Extensive experiments on benchmark datasets demonstrate that the proposed TLEE can significantly outperform the state-of-the-art approaches in terms of computational cost and inference latency, while maintaining competitive recognition accuracy. In addition, we verify the superiority of TLEE on the typical edge device NVIDIA Jetson Nano.

> 随着视频流的爆炸式增长，对高效可扩展视频理解的需求日益迫切。基于卷积神经网络（CNN）的先进视频识别方法通过采用二维或三维CNN架构展现出优异性能。然而，海量数据、高资源需求与严苛延迟要求阻碍了这些方案在资源受限的物联网及边缘设备上的广泛应用。为解决此问题，我们提出新型框架TLEE，使输入样本在二维CNN主干网络上同时具备时间维度和层级维度的早退能力，以实现高效视频识别。TLEE包含三类模块：门控模块、分支模块与特征复用模块。门控模块确定输入视频从哪一帧终止逐帧计算，而分支模块确定输入帧从CNN主干网络的哪一层终止逐层计算。此外，基于从退出分支获取的帧序列累积特征，特征复用模块可生成有效的视频表征以实现更高效的预测。在基准数据集上的大量实验表明，所提出的TLEE在计算成本和推理延迟方面显著优于现有先进方法，同时保持具有竞争力的识别准确率。我们还在典型边缘设备NVIDIA Jetson Nano上验证了TLEE的优越性。

This work was published by IEEE IoTJ (IEEE Internet of Things Journal). Click [here](https://doi.org/10.1109/JIOT.2023.3293506) for our paper.

## Required software

- Python 3.6+
- PyTorch (CUDA-enabled build recommended)
- torchvision
- NumPy
- Pandas
- Pillow
- PyYAML
- loguru

## Project Structure

```
TLEE/
├── arch/ # 2D CNN backbones used by TLEE
│ ├── VGG16.py # VGG-16 backbone with feature splits
│ ├── BN_Inception.py
│ ├── efficientnet.py
│ ├── mobilenetv2.py
│ └── mobilenetv2_tsm.py
├── config/ # YAML configuration files
│ ├── ucf101.yml / ucf101_branch.yml / ucf101_train.yml
│ ├── cloud_hmdb51_*_branch.yml / cloud_ucf101_*_branch.yml
│ └── cloud_ssv2_branch.yml
├── dataset/ # Video dataset loading & transforms
│ ├── dataset.py
│ └── transform.py
├── model/ # Core TLEE components
│ ├── tlee.py # TLEEModel & BaseModel definitions
│ ├── branch.py # Conv_Branch / FC_Branch classifiers
│ ├── branch_model.py # Auxiliary branch for early-exit decision
│ └── model_splits.py # Backbone splitting at intermediate layers
├── tools/ # Training / inference utilities
│ ├── FRM.py # Feature Reuse Module (Attention / EMA / Avg / Naive)
│ ├── random_exit.py # Random temporal / spatial exit baseline
│ └── lr_schedule.py # Step learning rate schedule
├── utils/ # Miscellaneous helpers
│ ├── config.py # YAML config wrapper
│ ├── compute_ap.py
│ ├── metrics.py
│ ├── misc.py
│ └── utils.py
├── log/
│ └── _log.py
├── env_config/ # Conda environment spec
├── main.py # Main entry: train/eval with branch heads
├── train_basemodel.py # Train a plain backbone (no early exit)
├── test.py # Quick CPU latency baseline
├── time_test.py # End-to-end latency / MACs evaluation
├── test.sh / cloud_*.sh # Job submission scripts
└── README.md
```

## Core Modules

### TLEEModel (`model/tlee.py`)

The full TLEE network that wraps a 2D CNN backbone with temporal gating and multiple layer-wise branch classifiers. It produces per-frame branch predictions and aggregates them through the Feature Reuse Module (FRM).

| Attribute | Description |
|---|---|
| `backbone` | 2D CNN (VGG16 / BN-Inception / MobileNetV2 / EfficientNet) |
| `splits` | Cut points of the backbone features (e.g. `[14, 24, 34, -1]` for VGG16) |
| `branches` | `Conv_Branch` / `FC_Branch` classifiers attached to each cut |
| `crm_num` | Number of channel-reduction convs inside each branch |
| `branch_scale` | Channel scaling factor of the reduction block |
| `useGateFunc` | Whether the temporal gating module is enabled |
| `useFeatureFunc` | Whether the feature-reuse aggregation is enabled |

**Key methods:**

- `forward(frames)` — Run a video clip through the backbone, executing early exit at the first branch whose confidence exceeds a threshold, and aggregate exit features for the final video-level prediction.

- `get_policies(...)` — Helper that returns a list of exit policies (per-frame, per-layer) used by `main.py` to decide when to stop computation.

- `BaseModel` — Plain 2D-CNN baseline with the same backbone but no early-exit branches, used as a reference in ablation studies.

### Branch (`model/branch.py`, `model/branch_model.py`)

Auxiliary classifiers attached to intermediate backbone layers. A `Branch` first reduces channels with stacked 1×1 convs, then applies a spatial pooling followed by a fully-connected head that outputs class logits.

| Class | Description |
|---|---|
| `Conv_Branch(size, inplanes, outputdim)` | Conv-based head with 1×1 / 3×3 / 1×1 feature extraction |
| `FC_Branch(size, inplanes, outputdim)` | Lightweight FC head (Linear → BN → ReLU → Linear) |
| `Branch` (`branch_model.py`) | Full branch module with Channel Reduction Module (CRM) and pool head |

### FeatureReuseModule / FRM (`tools/FRM.py`)

Aggregates the features coming from exit branches of consecutive frames so that the network can form a strong video-level representation even when each frame exits early.

| Class | Description |
|---|---|
| `AttentionFRM` | Two-layer MLP with sigmoid that weighs current vs. accumulated features |
| `EMAFRM` | Exponential moving average over previous features, learnable β |
| `AveragePooling` | Simple mean of the two feature tensors |
| `NaiveAdd` | Sum of absolute values as a naive baseline |

The class instantiated in `model/tlee.py` is configurable via `config.model.useFeatureFunc`.

### Random Exit (`tools/random_exit.py`)

Baseline used in the ablation study: at each frame, randomly decide whether to exit spatially (layer-wise) and / or temporally. Returns two boolean flags `(spatial_exit, temporal_exit)`.

### ModelSplits (`model/model_splits.py`)

Cuts a torchvision-style backbone into a `Sequential` of feature blocks and exposes the (location, output map size, channel count) of each cut, so that branches can be attached to the right tensors.

| Backbone | `split_location` | `split_map_size` | `split_channels` |
|---|---|---|---|
| VGG16 | `[14, 24, 34, -1]` | `[56, 28, 14, 7]` | `[128, 256, 512, 512]` |
| BN-Inception | `['_block_1', '_block_3a', '_block_4a', '_block_5a']` | `[28, 14, 7, 1]` | `[192, 576, 1056, 1024]` |
| MobileNetV2 | (see `model_splits.py`) | last stage uses `mean(2).mean(3)` pooling | — |

### Data Pipeline (`dataset/`)

`dataset/dataset.py` provides `get_dataloader(c_data, batch_size)` for the supported datasets (UCF101, HMDB51, Something-Something V2, Mini-Kinetics, ActivityNet 1.3). Frame-level preprocessing is configured per YAML file under `config/`.

### Utilities (`utils/`, `tools/lr_schedule.py`)

- `utils/config.py` — `Config` wrapper that turns a YAML file (or hierarchy) into a Python namespace accessible as `config.<key>.<subkey>`.
- `utils/utils.py` — `AverageMeter`, `accuracy`, `accuracy_gate`, `save_checkpoint`, `compute_mac`, `print_exp_data`, etc.
- `tools/lr_schedule.py` — Multi-step learning rate decay triggered at the epochs specified by `config.model.schedule` (each step multiplies the LR by 0.1).

## Usage

```bash
# 1. Create a Python environment (example with conda)
conda env create -f env_config/fe.yaml
conda activate fe

# 2. Edit a config file to point to your dataset / checkpoint,
#    e.g. config/ucf101.yml:
#      data.path_frame:   /path/to/UCF101/jpg
#      data.path_split:   /path/to/ucf101_rgb_val_split_1.txt
#      data.path_label:   /path/to/ucf101_rgb_val_split_1.txt
#      data.path_classid: /path/to/UCF101/labels/classInd.txt
#      checkpoint.init:   /path/to/pretrained_backbone.pth

# 3. Train the TLEE model with branch heads
python -u main.py config/ucf101_branch.yml

# 4. Train a plain backbone baseline (no early exit) for ablation
python -u train_basemodel.py config/ucf101_train.yml

# 5. Test a trained TLEE checkpoint
python -u test.py        # uses config/test settings
python -u time_test.py   # reports latency and GMACs

# 6. (Cluster) submit the cloud_* scripts after editing the YAML inside
bash cloud_ucf101_branch.sh
bash cloud_test_branchmodel.sh
bash cloud_test_tse.sh
bash cloud_time_test.sh
```

**YAML quick reference** (`config/ucf101.yml`):

| Section | Key | Meaning |
|---|---|---|
| `data` | `name` | Dataset name (`ucf101`, `hmdb51`, `ssv2`, …) |
| `data` | `num_frames` | Frames sampled per video clip (e.g. `10`) |
| `data.preprocessing` | `resize`, `crop_center`, `normalize` | Standard ImageNet preprocessing |
| `model` | `backbone` | One of `vgg16`, `BN_Inception`, `mobilenetv2`, `efficientnet*` |
| `model` | `branch_num` | Number of layer-wise exit branches |
| `model` | `crm_num` | Channel-reduction blocks inside each branch |
| `model` | `branch_scale` | Channel scaling factor of the reduction block |
| `model` | `useFeatureFunc` | Enable / disable the Feature Reuse Module |
| `model` | `useGateFunc` | Enable / disable the temporal gating module |
| `checkpoint` | `init` | Path to pretrained backbone weights |

## Citation

If you find TLEE useful or relevant to your project and research, please kindly cite our paper:

```
@ARTICLE{10176276,
  author={Wang, Qingli and Fang, Weiwei and Xiong, Neal N.},
  journal={IEEE Internet of Things Journal},
  title={TLEE: Temporal-Wise and Layer-Wise Early Exiting Network for Efficient Video Recognition on Edge Devices},
  year={2024},
  volume={11},
  number={2},
  pages={2842-2854},
  doi={10.1109/JIOT.2023.3293506}
}
```

## For more

The same authors have published a related work on UAV-assisted mobile edge computing using Deep Deterministic Policy Gradient, see [UAV-DDPG](https://github.com/fangvv/UAV-DDPG) for your reference. You can also refer to their other repositories: [VN-MADDPG](https://github.com/fangvv/VN-MADDPG) and [MTACP (IMPALA)](https://github.com/fangvv/MTACP).

## Contact

Qingli Wang (20120418@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
