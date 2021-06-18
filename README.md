﻿# RetinaTrack Pytorch

The pytorch implement of the retina track  original paper link: <https://arxiv.org/abs/2003.13870>

我的知乎解答链接: <https://zhuanlan.zhihu.com/p/269571970>

### Network
![image](https://github.com/Hanson0910/RetinaTrack/blob/main/source/RetinaTrack.png)

## Requirements
- python3.7
- pytorch 1.5

## Update Log

- [2020-10-23] Build the RetinaTrack Network
- [2020-10-25] fix some bug in network
- [2020-10-25] Build the loss train and inference process ,to verify the validity of the network 
- by overfitting a labeled image

## Model
- [Baidu cloud disk (code: j8yg)](https://pan.baidu.com/s/1-nIc0UZh5Zl8IuSUPgmbJA)
- [Google Driver](https://drive.google.com/file/d/13p15qH4KhLDAmlj-S98Am4mxCzWscAUX/view?usp=sharing)
- It should be noted that the above model is only used to verify the effectiveness of the network,It is generated by over fitting an image, so it only works on that image

## Demo
    python inference.py

## Training
    python train.py

## TODO
- [x] build the loss with the network
- create the dataloader and buile the complete training process
- private the mult vehicle tracking trained model

## Result
![image](https://github.com/Hanson0910/RetinaTrack/blob/main/source/result_img.jpg)

## References

Appreciate the great work from the following repositories:

- [pytorch retinanet](https://github.com/gm19900510/Pytorch_Retina_License_Plate)
