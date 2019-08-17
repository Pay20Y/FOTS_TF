Fast Oriented Text Spotting with a Unified Networks
### Introduction
This is an implementation of [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/pdf/1801.01671.pdf)
### Install
+ Python3.5
+ tensorflow 1.12.0
+ OpenCV
>> pip install -r requirements.txt
### Model
SynthText 6-epochs pretrained model can be found here
### Train
>> python3 main_train.py --gpu_list='0' --learning_rate=0.0001 --train_stage=2 --training_data_dir='/path/to/your/training images/' training_gt_data_dir='/path/to/your/training annotations'
### Test
>> python3 main_test.py --gpu_list='0' --test_data_path=/path/to/testset/ --checkpoint_path=checkpoints/
### How to use
#### Train your own data
If your data is same as the ICDAR annotations you can train the network directly.
Or you can implement your own data loader (![reference](data_provider/ICDAR_loader.py)). You should reimplement the load_annotations function which return the text polygons coordinates (shape: 4*2), text tags and transcriptions.
Note, for 'Don't care' text text tags should be True and transcription should be [-1]
And finetuning your own data on the SynthText pretrained model is suggested:
>> python3 main_train.py --gpu_list='0' --learning_rate=0.0001 --train_stage=2 --training_data_dir='/path/to/your/training images/' training_gt_data_dir='/path/to/your/training annotations' --pretrained_model_path='pretrained/model/path/'
### Examples and Experiments
Coming soon
### TODO
- [ ] Experiments on ICDAR2015 & ICDAR2017
- [ ] Decode with lexicon or language model
### Reference
+ [EAST](https://github.com/argman/EAST)
+ [FOTS.Pytorch](https://github.com/jiangxiluning/FOTS.PyTorch)
Thanks for the authors!
