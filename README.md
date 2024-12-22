## [AAAI2025] Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection
[Link to our paper](https://arxiv.org/abs/2412.10115)
![image](https://github.com/user-attachments/assets/d2680f1c-250b-447c-acb9-1a2cad28cbda)

## Requirements
```
conda env create -f environment.yml
```

## Dataset Preparation

1. Download the original dataset MVTec, PACS and CIFAR-10.

2. Generate the corrupted test set for MVTec and CIFAR-10.

```
python generate_corrupted_mvtec.py

python generate_corrupted_cifar10.py
```

Arrange data with the following structure (e.g. MVTec dataset):
```
Path/To/Dataset
├── mvtec
      ├── bottle
      ├── ......
├── mvtec_brightness
      ├── bottle
      ├── ......
├── mvtec_contrast
      ├── bottle
      ├── ......
├── mvtec_defocus_blur
      ├── bottle
      ├── ......
├── mvtec_gaussian_noise
      ├── bottle
      ├── ......
```
Modify the file path in the scripts.

## Training
For the training process, please simply execute (e.g. MVTec dataset):
```
python train_mvtec_fico.py
```

## Inference
For the inference process, please simply execute (e.g. MVTec dataset):
```
python inference_mvtec_ATTA.py
```

## Acknowledgment
We thank the authors from [ADShift](https://github.com/mala-lab/ADShift) for reference. We modify their code to implement FiCo.

## Citation
```
@article{chen2024filter,
  title={Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection},
  author={Chen, Zining and Luo, Xingshuang and Wang, Weiqiu and Zhao, Zhicheng and Su, Fei and Men, Aidong},
  journal={arXiv preprint arXiv:2412.10115},
  year={2024}
}
```


