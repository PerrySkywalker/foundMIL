# When Multiple Instance Learning Meets Foundation Models: Advancing Histological Whole Slide Image Analysis

### The source code has evolved from TransMIL, ABMIL, RRT, WiKG, DTFD, CLAM.

### Now supports TransMIL, ABMIL, RRT, WiKG


### Train
```python
python train.py --stage='train' --config='config/ABMIL.yaml'
```
### Test
```python
python train.py --stage='test' --config='config/ABMIL.yaml'
```


###
```
lightning==2.0.9.post0
pytorch-lightning==2.0.9.post0
pytorch-toolbelt==0.6.3
torch==2.10
```
###