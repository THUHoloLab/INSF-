<div align="center">

# Dynamic Computational Spectral Imaging<br>with Implicit Neural Spectral Fields

**Jinwen Wei · Xinyu Liu · Shuhe Zhang · Liangcai Cao**

Department of Precision Instrument, Tsinghua University

[![License](https://img.shields.io/github/license/THUHoloLab/INSF?color=2ea44f)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

<br>

<img src="assets/insf.gif" width="100%" alt="How INSF works: coded acquisition and implicit neural spectral field">

</div>

<br>

**INSF** turns a single coded frame of a filter-array camera into a full spectral cube.
The scene is an implicit neural spectral field — a small coordinate network that maps
*(x, y, λ)* to intensity — fitted directly to the measurement, with **no training data**.
Sharing the field across frames, aligned by optical flow, extends it to **moving scenes**,
and the compact representation keeps a 98-frame spectral video at about **1 GB instead of 62 GB**.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128          # match your CUDA
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .

python main.py                                                                   # list all commands
python main.py reconstruct shared-field --config configs/move1_shared.yaml       # static scene
python main.py reconstruct mc-window   --config configs/move2_mc_window.yaml     # moving scene
```

Camera data are not included; see [`data/README.md`](data/README.md).

## Citation

```bibtex
@article{wei2026insf,
  title   = {Dynamic Computational Spectral Imaging with Implicit Neural Spectral Fields},
  author  = {Wei, Jinwen and Liu, Xinyu and Zhang, Shuhe and Cao, Liangcai},
  year    = {2026}
}
```
