# Multi-CycleGAN

**Multi Cycle Consistent Adversarial Networks for Multi Attribute Image to Image Translation**
Seokhee Jo, Kyu Cheol Cho — *Journal of The Korea Society of Computer and Information*, Vol. 25 No. 9, pp. 63–69, 2020
[[Paper (PDF)](multicycleGAN.pdf)] [[DOI: 10.9708/jksci.2020.25.09.063](https://doi.org/10.9708/jksci.2020.25.09.063)]

<p align="center"><img src="https://user-images.githubusercontent.com/33644885/114181845-fa580900-997c-11eb-88bc-bd9065548c67.png" width="80%"></p>

## Idea

Standard CycleGAN translates between **one** source and **one** target attribute. Real images usually change along several attributes at once (e.g. *blonde* + *female*), and data for the combined target domain is often scarce.

Multi-CycleGAN learns two attributes **separately** and composes them:

- Three domains **A** (source), **B** (attribute 1), **C** (attribute 2) are trained with two-way cycle consistency.
- Translation chain: `A → B → C` via `G_A`, `G_C`, with reverse generators `G_B`, `G_R` closing each cycle.
- Because attributes are learned from separate domains, the model can produce combinations that rarely appear together in training data (e.g. *nature* + *night*), while preserving more of the input's structure.

<p align="center"><img src="https://user-images.githubusercontent.com/33644885/114182018-25daf380-997d-11eb-8742-2a85e6ea5340.PNG" width="80%"></p>

### Losses
GAN losses for `D_A`, `D_B`, `D_C`, and cycle losses `cycle_A`, `cycle_B`, `cycle_B2`, `cycle_C` (weights `--lambda_A/B/B2/C`), plus identity loss.
Implementation: [`models/develop_model.py`](models/develop_model.py).

## Usage

```bash
pip install -r requirements.txt
```

Dataset layout (unaligned):
```
datasets/<name>/trainA   # source domain
datasets/<name>/trainB   # attribute 1
datasets/<name>/trainC   # attribute 2
```

Train / test:
```bash
python train.py --dataroot ./datasets/<name> --name <exp_name> --model develop
python test.py  --dataroot ./datasets/<name> --name <exp_name> --model develop
```

## Limitations
- Four generators and three discriminators → many parameters and slow training.
- Sensitive to small changes in the input, so failure cases appear more easily than with single-attribute CycleGAN.

## Acknowledgements
Built on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (BSD license; see [LICENSE](LICENSE)).
