# Full_TT_Net

## Requirement
Python3.6

Pytorch 1.4.0

## Run the train.py script
```python
python3 train.py --arch important_sketching_ftt_multi_relu
```
Architechture:Cifar10 ---> Important Sketching ---> TT Net

## Run the cov_input_train.py script
```python
python3 cov_input_train.py 
```
Architechture:Pretrained LeNet(untrainable) ---> Covariates ---> Important Sektching ---> WideResNet 

## HyperParameter.
1.settings.ALPHA: penaty of not being orthogonal

2.settings.LR: learning rate

