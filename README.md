<!--
 * @Author: Jiayi Liu
 * @Date: 2022-10-02 08:25:41
 * @LastEditors: Jiayi Liu
 * @LastEditTime: 2023-05-25 03:39:09
 * @FilePath: /private_jacieliu/DL-paper-implementation/README.md
 * @Description: 
 * Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
-->
Paper得来终觉浅，绝知此事要coding。

Knowledge obtained on the papers always feels shallow, and it must be known that this thing requires coding.

## Purpose

1. Minimal Practice
2. Project Notes
3. Optimization
4. Algorithm Competition

## Basic

### 1. CNN

| Model   | Link   | Paper  | Code  |
| ----  | ----  | ----  | ----  |
| Resnet  | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)  | :white_check_mark: | :white_check_mark: |
| InceptionV3  | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)  | :white_check_mark: | :white_check_mark: |
| InceptionV4  | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)  | :white_check_mark: | :white_large_square: |
| MobileNet  | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  | :white_large_square: | :white_large_square: |
| EfficientNet  | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)  | :white_large_square: | :white_large_square: |
| Residual Attention Network  | [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)  | :white_check_mark: | :white_check_mark: |
| Non-deep Networks  | [Non-deep Networks](https://arxiv.org/abs/2110.07641)  | :white_large_square: | :white_large_square: |

### 2. RNN
| Model   | Link   | Paper  | Code  |
| ----  | ----  | ----  | ----  |
| LSTM  | [Long Short-term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)  | :white_check_mark: | :white_check_mark: |
| BiLSTM  | [Bidirectional recurrent neural networks](https://www.researchgate.net/publication/3316656_Bidirectional_recurrent_neural_networks)  | :white_check_mark: | :white_large_square: |
| GRU  | [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)  | :white_check_mark: | :white_large_square: |

### 3. Transformer

| Model   | Link   | Paper  | Code  |
| ----  | ----  | ----  | ----  |
| Transformer  | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  | :white_check_mark: | :white_check_mark: |
| BERT  | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  | :white_check_mark: | :white_large_square: |
| GPT-3  | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)  | :white_check_mark: | :white_large_square: |
| ViT  | [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929)  | :white_check_mark: | :white_large_square: |


### 4. Generation

| Model   | Link   | Paper  | Code  |
| ----  | ----  | ----  | ----  |
| GAN  | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)  | :white_check_mark: | :white_large_square: |
| pix2pix  | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)  | :white_large_square: | :white_large_square: |
| CycleGAN  | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)  | :white_check_mark: | :white_large_square: |
| VAE  | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)  | :white_check_mark: | :white_check_mark: |
| DDPM  | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  | :white_check_mark: | :white_large_square: |
| Guided Diffusion  | [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)  | :white_check_mark: | :white_large_square: |
| DALL.E 2  | [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)  | :white_large_square: | :white_large_square: |

### 5. Multimodal

| Model   | Link   | Paper  | Code  |
| ----  | ----  | ----  | ----  |
| CLIP  | [Learning Transferable Visual Models From Natural Language Supervision(Connecting Text and Images)](https://arxiv.org/abs/2103.00020)  | :white_check_mark: | :white_check_mark: |
| ViLT  | [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)  | :white_check_mark: | :white_large_square: |
| SimVLM  | [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/abs/2108.10904)  | :white_check_mark: | :white_large_square: |
| ALBEF  | [Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://proceedings.neurips.cc/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html)  | :white_check_mark: | :white_check_mark: |
| VLMo  | [VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d46662aa53e78a62afd980a29e0c37ed-Abstract-Conference.html)  | :white_check_mark: | :white_large_square: |
| BLIP  | [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n.html)  | :white_check_mark: | :white_large_square: |
| CYCLIP  | [CyCLIP: Cyclic Contrastive Language-Image Pretraining](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2cd36d327f33d47b372d4711edd08de0-Abstract-Conference.html)  | :white_check_mark: | :white_large_square: |
| +MAE  | [Training Vision-Language Transformers from Captions Alone](https://arxiv.org/abs/2205.09256)  | :white_check_mark: | :white_large_square: |
| VLMixer  | [VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix](https://proceedings.mlr.press/v162/wang22h.html)  | :white_check_mark: | :white_large_square: |


## Project
### 1. Object Detection

| Model   | Link   | Paper  | Code  |
| ----  | ----  | ----  | ----  |
| R-CNN  | [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)  | :white_check_mark: | :white_large_square: |
| Faster R-CNN  | [Faster R-CNN](https://arxiv.org/abs/1504.08083)  | :white_check_mark: | :white_large_square: |
| YoloV3  | [You Only Look Once: Unified, Real-time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)  | :white_check_mark: | :white_large_square: |
| DETR  | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)  | :white_large_square: | :white_large_square: |

### 3. Audio-visual

| Model   | Link   | Paper  | Code  |
| ----  | ----  | ----  | ----  |
| Syncnet  | [Out of time: automated lip sync in the wild](https://link.springer.com/chapter/10.1007/978-3-319-54427-4_19)  | :white_check_mark: | :white_large_square: |
| Wav2lip  | [A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild](https://arxiv.org/abs/2008.10010)  | :white_check_mark: | :white_check_mark: |