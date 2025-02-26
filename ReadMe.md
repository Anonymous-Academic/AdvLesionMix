

# AdvLesionMix


### Environment

This source code was tested in the following environment:

    Python 3.8.19
    PyTorch: 1.11.0+cu113
    Torchvision: 0.12.0+cu113
    Timm: 1.0.7

### Dataset

* (1) Download the ISIC 2017 and ISIC 2018 datasets and organize the structure as follows:
```
dataset folder
├── train
│   ├── class_001
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── class_002
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── ...
├── validation
│   ├── class_001
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── class_002
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── ...
└── test
    ├── class_001
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    ├── class_002
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── ...

```
* (2) Resize all images to 256x256. You may consider using the `resize_dataset.py` script we provide.
* (3) modify the path in `utils.py` to the dataset folders.

### Training
* Extract the pre-trained CycleGAN weights:
```
cd CycleGan_weights/weights
cat netG_A2B_part_* > netG_A2B.tar.gz
tar -xzvf netG_A2B.tar.gz
cd ..
cd ..
```
* For baselines:
```
chmod +x run_baseline.sh
./run_baseline.sh
```
* For the proposed approach:
```
chmod +x run_proposed.sh
./run_proposed.sh
```
