# AM-Net
This is the implementation code for the paper, <a href="https://ieeexplore.ieee.org/abstract/document/10123114"> "An Attention-guided Multistream Feature Fusion Network for Early Localization of Risky Traffic Agents in Driving Videoss"</a>, <i> IEEE Transaction on Intelligent Vehicles,</i> 2023.</p>


The objective of this project is to determine a riskiness score for all traffic agents within a driving scene. In other words, the goal of this paper is to achieve early localization of potentially risky traffic agents in driving videos.

<a name="dataset"></a>
## Dataset Preparation
The code currently supports ROL dataset:
> * Please refer to the [ROL Official](https://github.com/monjurulkarim/ROL_Dataset) repo for downloading and deployment. 

<a name="install"></a>
## Installation Guide
### 1. Setup Python Environment

The code is implemented and tested with `Python=3.7.9` and `PyTorch=1.2.0` with `CUDA=10.2`. We highly recommend using Anaconda to create virtual environment to run this code.


After setting up the python environment, please use the following command to verify that everything is correctly configured:
```shell
python main.py --phase=check
```
### 2. Train and Test
To train the AM-Net model using ROL dataset, please ensure that you have downloaded and placed the extracted CNN features of ROL dataset in the directory. Then, run the following command:
```shell
python main.py --phase=train
```

Use the following command to test the trained model:
```shell
python main.py --phase=test
```

To save the riskiness score of the traffic agents use the following command:
```shell
python demo.py
```

### 3.  Pre-trained Models
> * [**Pre-trained AM-Net Models**](https://drive.google.com/drive/folders/1zv_1h8zBocywhU5fsPeKtbxTp7xlMZYL?usp=sharing): You can also download and use the pre-trained weights of AM-Net. The pretrained model weights are intended for testing and demo purposes. To utilize the pre-trained model for testing, please download it and ensure it is placed in the appropriate directory.


<a name="citation"></a>
## Citation

Please cite our paper if you find the code useful.

```
@ARTICLE{karim_am_net2023,
  author={Karim, Muhammad Monjurul and Yin, Zhaozheng and Qin, Ruwen},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={An Attention-guided Multistream Feature Fusion Network for Early Localization of Risky Traffic Agents in Driving Videos}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TIV.2023.3275543}}
```

