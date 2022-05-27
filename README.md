# Object Detection in Aerial Images: A Case Study on Performance Improvement

Object Detection (OD) in aerial images has gained much attention due to its applications in search and rescue, town planning, and agriculture yield prediction etc. Recently introduced large-scale aerial images dataset, iSAID has enabled the researchers to advance the OD tasks on satellite images. Unfortunately, the available OD pipelines and ready-to-train architectures are well-tailored and configured to be used with tasks dealing with natural images. In this work, we study that directly using the available object detectors, specifically the vanilla Faster RCNN with FPN is sub-optimal for aerial OD. To help improve its performance, we tailor the Faster R-CNN architecture and propose several modifications including changes in architecture in different blocks of detector, training \& transfer learning strategies, loss formulations, and other pre-post processing techniques. By adopting the proposed modifications on top of the vanilla Faster-RCNN, we push the performance of the model and achieve an absolute gain of <b> 4.44 AP </b> over the vanilla Faster R-CNN on the iSAID validation set.

This repository contains the code files for reproducing the main experiments mentioned in our paper. Moreover, this repo supports the use of third party backbones to be integrated with the Faster R-CNN object detectors including SWIN, ConvNext and timm backbones.

## Technical Report 
Complete technical report can be viewed [here](https://github.com/MUKhattak/OD-Satellite-iSAID/blob/OD_SatteliteImages/projects/OD_satellite_iSAID/technical_report.pdf).

## Requirements and Installation
We have tested this code on Ubuntu 20.04 LTS with Python 3.8. This repo is heavly built on Detectron2. Follow the instructions below to setup the environment and install the dependencies.
 ```shell
 conda create -n detectron_OD python=3.8
 conda activate detectron_OD
 # Install torch and torchvision
 pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
 # Install Detectron2 (for more details visit : https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
 python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
 # Install matplotlib for visualizations
 python -m pip install -U matplotlib
 ```
 
 Now clone this repository:
  ```shell
 git clone https://github.com/MUKhattak/OD-Satellite-iSAID.git
 cd OD-Satellite-iSAID/
```

## Command-line parameters
| Parameter          | Discription                                                                                                                 |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|
| config-file        | path to the config file which you want to run e.g --config-file ./configs/derived_configs/fastrcnn_timm_backbone.yaml       |
| isaid-path         | path to the iSAID dataset folder                                                                                            |

## Dataset
We use iSAID dataset: A Large-scale Dataset for Instance Segmentation in Aerial Images[1]. Our code expects the dataset folder to have the following structure,

```
isaid_dataset_root-folder/
└─ train
    ├─ images
        ├─ P1207_1800_2600_1200_2000.png
        ├─ P1207_1800_2600_1800_2600.png
        ├─ .......
    ├─ instancesonly_filtered_train.json
└─ val
    ├─ images
        ├─ P1557_3000_3800_0_800.png
        ├─ .......
    ├─ instancesonly_filtered_val.json
```



## Training and Evaluation  

### Training
We implement our code as seperate project in detectron2, so first `cd` to the project folder by running the following commad.

  ```bash
 $ cd projects/OD_satellite_iSAID/
```
#### Using different backbones
To train a vanilla Faster R-CNN with FPN-R101 backbone, run the following command

  ```bash
 $ python plain_train_net.py --config-file ./configs/derived_configs/faster_rcnn_R_101_FPN_3x.yaml --isaid-path /path/to/isaid/root/folder
```

To train a Faster R-CNN with ConvNext-Base backbone, run the following command

  ```bash
 $ python plain_train_net.py --config-file ./configs/derived_configs/faster_rcnn_convnext_base_FPN_3x.yaml --isaid-path /path/to/isaid/root/folder
```
To train a Faster R-CNN with SWIN-Base backbone, run the following command

  ```bash
 $ python plain_train_net.py --config-file ./configs/derived_configs/faster_rcnn_swin_base_3x_FPN.yaml --isaid-path /path/to/isaid/root/folder
```
To train a Faster R-CNN with timm-based ResNet-BiT backbone, run the following command

  ```bash
 $ python plain_train_net.py --config-file ./configs/derived_configs/fastrcnn_timm_backbone.yaml --isaid-path /path/to/isaid/root/folder
```
Note: To use any other backbone from timm (pytorch image models), you may need to change slightly change the config file and ```timm_backbone.py```[here]([https://github.com/MUKhattak/OD-Satellite-iSAID/blob/OD_SatteliteImages/projects/OD_satellite_iSAID/technical_report.pdf](https://github.com/MUKhattak/OD-Satellite-iSAID/blob/OD_SatteliteImages/projects/OD_satellite_iSAID/detectron2/modeling/backbone/timm_backbone.py)) depending on the number of feature maps that can be obtained from the respective timm backbone.

To train a Faster R-CNN with Deformable ResNet-101 backbone,run the following command

  ```bash
 $ python plain_train_net.py --config-file ./configs/derived_configs/faster_rcnn_deformable_resnet101.yaml --isaid-path /path/to/isaid/root/folder
```

#### Using different loss functions
To train a Faster R-CNN FPN-R101 with federated loss (+ sigmoid cross-entropy) , run the following command

  ```bash
 $ python plain_train_net.py --config-file ./configs/derived_configs/faster_rcnn_fed_loss.yaml --isaid-path /path/to/isaid/root/folder
```

To train a Faster R-CNN FPN-R101 with focal loss, run the following command

  ```bash
 $ python plain_train_net.py --config-file ./configs/derived_configs/faster_rcnn_focal_loss.yaml --isaid-path /path/to/isaid/root/folder
```
We have also provided additional config files which can be explored at ```projects/OD_satellite_iSAID/configs/derived_configs```

#### Combining different components in a single config file
You can use the different configurations together (e.g using specific backbone and loss function together) by merging the respective config files, and use the new config file for training.

#### Other modifications
We also explore other architectural and training modifications in the vanilla Faster R-CNN detector, please refer to our [report](https://github.com/MUKhattak/OD-Satellite-iSAID/blob/OD_SatteliteImages/projects/OD_satellite_iSAID/technical_report.pdf) and [base config](https://github.com/MUKhattak/OD-Satellite-iSAID/blob/OD_SatteliteImages/projects/OD_satellite_iSAID/configs/Base-RCNN-FPN.yaml) file for additional details.

### Evaluation
To evaluate a trained model on the iSAID validation set, run the following command
  ```bash
 $ python plain_train_net.py --config-file ./path/to/custom/config/file --eval-only --isaid-path /path/to/isaid/root/folder
```

## Grounth-truth and prediction visualizations
We also provide a jupyter notebook for visualizing the GT and predicted bounding boxes on respective images for qualitative results.
Please refer to this path to view the notebook ```projects/OD_satellite_iSAID/Detectron2_Faster-RCNN_iSAID.ipynb``` (or click [here](https://github.com/MUKhattak/OD-Satellite-iSAID/blob/OD_SatteliteImages/projects/OD_satellite_iSAID/Detectron2_Faster-RCNN_iSAID.ipynb)).

