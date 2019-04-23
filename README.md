# Fast Oriented Text Spotting with a Unified Networkt
### Introduction
This is an implementation of [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/pdf/1801.01671.pdf)
### Install
+ Python2
+ tensorflow
+ OpenCV
### Model
Later
### Train
>> python2 multigpu_train.py --gpu_list=gpu_id --training_data_path=/path/to/trainset/
You should also change line 824 in icdar.py should be changed for the path of annotation file
### Test
>> python2 eval.py --gpu_list=gpu_id --test_data_path=/path/to/testset/ --checkpoint_path=checkpoints/
### Examples
![image_1](demo_images/img_1.jpg)
![image_2](demo_images/img_2.jpg)
![image_3](demo_images/img_3.jpg)
### Differences from paper
Without OHEM
Pretrained on Synth800k for 6 epochs
Fine-tuned on ICDAR15 only without ICDAR2017 MLT
And it can only get F-score 41 on ICDAR2015 testset, more training tricks is needed
### Reference
[EAST](https://github.com/argman/EAST)
[FOTS.Pytorch](https://github.com/jiangxiluning/FOTS.PyTorch)
