## TLEE

This is the source code for our paper: **TLEE: Temporal-wise and Layer-wise Early Exiting Network for Efficient Video Recognition on Edge Devices**. A brief introduction of this work is as follows:

> With the explosive growth in video streaming comes a rising demand for efficient and scalable video understanding. State-of-the-art video recognition approaches based on Convolutional Neural Network (CNN) have shown promising performance by adopting 2D or 3D CNN architectures. However, the large data volumes, high resource demands, and strict latency requirements have hindered the wide application of these solutions on resource-constrained Internet-of-Things (IoT) and edge devices. To address this issue, we propose a novel framework called TLEE that enables the input samples the abilities of both Temporal-wise and Layer-wise Early Exiting on 2D CNN backbones for efficient video recognition. TLEE consists of three types of modules: gating module, branch module, and feature reuse module. The gating module determines for an input video from which frame of this video to exit the per-frame computation, while the branch module determines for an input frame from which layer of the CNN backbone to exit the per-layer computation. Besides, based on the accumulated features of frame sequences from exit branches, the feature reuse module generates effective video representations to enable more efficient predictions. Extensive experiments on benchmark datasets demonstrate that the proposed TLEE can significantly outperform the state-of-the-art approaches in terms of computational cost and inference latency, while maintaining competitive recognition accuracy. In addition, we verify the superiority of TLEE on the typical edge device NVIDIA Jetson Nano.

> 随着视频流的爆炸式增长，对高效可扩展视频理解的需求日益迫切。基于卷积神经网络（CNN）的先进视频识别方法通过采用二维或三维CNN架构展现出优异性能。然而，海量数据、高资源需求与严苛延迟要求阻碍了这些方案在资源受限的物联网及边缘设备上的广泛应用。为解决此问题，我们提出新型框架TLEE，使输入样本在二维CNN主干网络上同时具备时间维度和层级维度的早退能力，以实现高效视频识别。TLEE包含三类模块：门控模块、分支模块与特征复用模块。门控模块确定输入视频从哪一帧终止逐帧计算，而分支模块确定输入帧从CNN主干网络的哪一层终止逐层计算。此外，基于从退出分支获取的帧序列累积特征，特征复用模块可生成有效的视频表征以实现更高效的预测。在基准数据集上的大量实验表明，所提出的TLEE在计算成本和推理延迟方面显著优于现有先进方法，同时保持具有竞争力的识别准确率。我们还在典型边缘设备NVIDIA Jetson Nano上验证了TLEE的优越性。

This work was published by IEEE IoTJ (IEEE Internet of Things Journal). Click [here](https://doi.org/10.1109/JIOT.2023.3293506) for our paper.

## Required software

PyTorch

## Citation
    @ARTICLE{10176276,
    author={Wang, Qingli and Fang, Weiwei and Xiong, Neal N.},
    journal={IEEE Internet of Things Journal}, 
    title={TLEE: Temporal-Wise and Layer-Wise Early Exiting Network for Efficient Video Recognition on Edge Devices}, 
	year={2024},
	volume={11},
	number={2},
	pages={2842-2854},
	doi={10.1109/JIOT.2023.3293506}}


## Contact

Qingli Wang (20120418@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
