## TLEE

This is the source code for our paper: **TLEE: Temporal-wise and Layer-wise Early Exiting Network for Efficient Video Recognition on Edge Devices**. A brief introduction of this work is as follows:

> With the explosive growth in video streaming comes a rising demand for efficient and scalable video understanding. State-of-the-art video recognition approaches based on Convolutional Neural Network (CNN) have shown promising performance by adopting 2D or 3D CNN architectures. However, the large data volumes, high resource demands, and strict latency requirements have hindered the wide application of these solutions on resource-constrained Internet-of-Things (IoT) and edge devices. To address this issue, we propose a novel framework called TLEE that enables the input samples the abilities of both Temporal-wise and Layer-wise Early Exiting on 2D CNN backbones for efficient video recognition. TLEE consists of three types of modules: gating module, branch module, and feature reuse module. The gating module determines for an input video from which frame of this video to exit the per-frame computation, while the branch module determines for an input frame from which layer of the CNN backbone to exit the per-layer computation. Besides, based on the accumulated features of frame sequences from exit branches, the feature reuse module generates effective video representations to enable more efficient predictions. Extensive experiments on benchmark datasets demonstrate that the proposed TLEE can significantly outperform the state-of-the-art approaches in terms of computational cost and inference latency, while maintaining competitive recognition accuracy. In addition, we verify the superiority of TLEE on the typical edge device NVIDIA Jetson Nano.

> TLEE：用于边缘设备上高效视频识别的时间和层早期退出网络

This work will be published by IEEE IoTJ (IEEE Internet of Things Journal). Click [here](https://doi.org/10.1109/JIOT.2023.3293506) for our paper.

## Required software

PyTorch

## Citation
    @ARTICLE{10176276,
    author={Wang, Qingli and Fang, Weiwei and Xiong, Neal N.},
    journal={IEEE Internet of Things Journal},
    title={TLEE: Temporal-wise and Layer-wise Early Exiting Network for Efficient Video Recognition on Edge Devices},
    year={2023},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/JIOT.2023.3293506}}


## Contact

Qingli Wang (20120418@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
