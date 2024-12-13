# CrackCLIP：Adapting Vision-Language Models for Weakly Supervised Crack Segmentation

A Pytorch implementation of CLIP-based Weakly Supervised Crack Segmentaion projects.
## Datasets：Crack500, CrackForest, DeepCrack.
Notes：please download the corresponding dataset and prepare it by following the [guidance](https://pan.baidu.com/s/1lz910hv81JSPq87-rN9RGQ?pwd=esjm) code: esjm
## Installation： You can create a new Conda environment using the command:
```
conda create -n my_pytorch python=3.7
source activate my_pytorch
pip install -r requirements.txt
```

## Training
Before the training, please download the dataset and copy it into the folder "data".
```
--data
----crack
------pavement
--------test
----------CFD
----------crack500
----------DeepCrack
--------train
----------crack500
------------image
------------mask
```
Check the dataset setting of CrackCLIP training in file datasets.py.
Training CrackCLIP model by `python train.py` or `sh train.sh`.

## Testing
Testing CrackCLIP model by `python test.py`




