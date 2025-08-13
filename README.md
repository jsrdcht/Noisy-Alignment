# Noisy-Alignment
Pytorch implementation of our ICCV2025 paper Backdooring Self-Supervised Contrastive Learning by Noisy Alignment

# Training
We follow Saha et al.'s implementation [UMBCvision/SSL-Backdoor](https://github.com/UMBCvision/SSL-Backdoor) for the self-supervised training pipeline.

Poison synthesis uses the core function `add_watermark` defined in `Noisy-Alignment/utils.py`.

```python
from utils import add_watermark

# patch trigger (random location by default)
img = add_watermark(img, watermark, mode='patch')

# blended trigger (global alpha blending)
img = add_watermark(img, watermark, mode='blend')
```

# Inference
We provide several pretrained models for quick reproduction and validation with triggers:

- CIFAR-10 pretrained (SimSiam): `simsiam_class_airplane_trigger_HTBA14.pth.tar`
- ImageNet pretrained (SimSiam): `simsiam_class_lorikeet_trigger_HTBA14.pth.tar`

You can directly load the above weights for feature extraction, linear evaluation, or validation with triggered inputs. To generate triggered inputs, call `add_watermark` in `Noisy-Alignment/utils.py` and feed the resulting images to the model.

# Citation
```bibtex
@inproceedings{chen2025backdooring,
  title={Backdooring Self-Supervised Contrastive Learning by Noisy Alignment},
  author={Tuo Chen, Jie Gui, Minjing Dong, Lanting Fang, Ju Jia, Jian Liu},
  journal={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```