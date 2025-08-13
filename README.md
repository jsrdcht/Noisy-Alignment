# Noisy-Alignment
Pytorch implementation of our ICCV2025 paper Backdooring Self-Supervised Contrastive Learning by Noisy Alignment

# Training
We follow Saha et al.'s implementation [UMBCvision/SSL-Backdoor](https://github.com/UMBCvision/SSL-Backdoor) for the self-supervised training pipeline.

Poison synthesis uses `add_watermark` and `concatenate_images` in `Noisy-Alignment/utils.py`.
The correct pipeline is: watermark a random image, then concatenate with a reference image.

```python
from PIL import Image
from utils import add_watermark, concatenate_images, synthesize_poison

# Inputs
random_img = Image.open('random.jpg').convert('RGB')
reference_img = Image.open('reference.jpg').convert('RGB')
watermark = Image.open('wm.png').convert('RGBA')

# Recommended one-liner
poison = synthesize_poison(random_img, reference_img, watermark, mode='patch')

# Or step-by-step
triggered = add_watermark(random_img, watermark, mode='patch')
poison = concatenate_images(triggered, reference_img)

# Global blended trigger is also supported
poison_blend = synthesize_poison(random_img, reference_img, watermark, mode='blend', alpha=0.2)
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