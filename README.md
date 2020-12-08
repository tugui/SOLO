# SOLO
The code is an unofficial pytorch implementation of [SOLO: Segmenting Objects by Locations](https://arxiv.org/abs/1912.04488) in mmdetection v2.7.0.

## Install
The code is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [get_start.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation instructions.

## Training
Follows the same way as mmdetection.

```
python tools/train.py configs/solo/solo_r50_fpn_8gpu_1x.py

python tools/train.py configs/solo/solo_r50_fpn_8gpu_3x.py

python tools/train.py configs/solo/solo_r101_fpn_8gpu_3x.py
```

## Reference
- [Official Code](https://github.com/WXinlong/SOLO)
- [Unofficial Code](https://github.com/Epiphqny/SOLO)

## TODO
- Decoupled SOLO
- SOLOv2