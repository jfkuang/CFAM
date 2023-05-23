Code and dataset for Visual Information Extraction in the Wild: Practical Dataset and End-to-end Solution. (ICDAR2023)

POIE dataset is available at https://drive.google.com/file/d/1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs/view?usp=share_link.

More details on the code and dataset will be refined soon. Thank you for your attention.

## Introduction

This project is about a novel end-to-end framework with a plug-and-play CFAM for VIE tasks, which adopts contrastive learning and properly designs the representation of VIE tasks for contrastive learning.  

The main branch works with **PyTorch 1.6+**.

<!-- <div align="center">
  <img src="https://user-images.githubusercontent.com/24622904/187838618-1fdc61c0-2d46-49f9-8502-976ffdf01f28.png"/>
</div> -->

### Major Features

<!-- - **Comprehensive Pipeline**

  The toolbox supports not only text detection and text recognition, but also their downstream tasks such as key information extraction.

- **Multiple Models**

  The toolbox supports a wide variety of state-of-the-art models for text detection, text recognition and key information extraction.

- **Modular Design**

  The modular design of MMOCR enables users to define their own optimizers, data preprocessors, and model components such as backbones, necks and heads as well as losses. Please refer to [Overview](https://mmocr.readthedocs.io/en/dev-1.x/get_started/overview.html) for how to construct a customized model.

- **Numerous Utilities**

  The toolbox provides a comprehensive set of utilities which can help users assess the performance of models. It includes visualizers which allow visualization of images, ground truths as well as predicted bounding boxes, and a validation tool for evaluating checkpoints during training.  It also includes data converters to demonstrate how to convert your own data to the annotation files which the toolbox supports. -->

## Installation

MMOCR depends on [PyTorch](https://pytorch.org/), [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
git clone https://github.com/jfkuang/CFAM.git
cd mmocr
mim install -e .
```


## Acknowledgement

We appreciate MMOCR as our codebase.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{
    title={Visual Information Extraction in the Wild: Practical Dataset and End-to-end Solution},
    author={Jianfeng Kuang, Wei Hua, Dingkang Liang, Mingkun Yang, Deqiang Jiang, Bo Ren, Yu Zhou, Xiang Bai},
    journal= {The 17th International Conference on Document Analysis and Recognition},
    year={2023}
}
