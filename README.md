# INSF: Implicit Neural Spectral Fields

Code for *Dynamic Computational Spectral Imaging with Implicit Neural Spectral Fields*
(Jinwen Wei, Xinyu Liu, Shuhe Zhang, Liangcai Cao).

INSF reconstructs a 601-band spectral datacube (350–950 nm) from coded measurements of a
filter-array spectral camera.  The scene is an implicit neural representation that is optimized
directly on the measurements of each scene, without pretraining or training data:

```
spatial features   H(x, y)            2D multiresolution hash grid (tiny-cuda-nn)
spectral FiLM      γ(λ), δ(λ)          48 learnable knots, linear interpolation
field              X(x, y, λ) = c · softplus( MLP( γ(λ) ⊙ H(x, y) + δ(λ) ) )
forward model      Y(x, y)    = Σ_λ M(x, y, λ) X(x, y, λ)
```

## Layout

```
INSF/
├── main.py              entry point: python main.py <group> <command> [--config file.yaml] [arguments]
├── pyproject.toml       pip install -e .   (also installs the `insf` command)
├── configs/             one YAML file per experiment of the paper
├── insf/
│   ├── models/          spatial–spectral FiLM field and its space–time variant
│   ├── forward.py       forward-model helpers, Charbonnier loss, learning-rate schedule, rendering
│   ├── recon/           single-frame, shared-field, mean-input, optical flow, motion-compensated multiframe
│   ├── baselines/       TwIST, ADMM, MST / MST++
│   ├── data/            mask and measurement I/O, raw-frame conversion, held-out pixels, colour-chart frames
│   ├── simulation/      coded measurements of public datasets, controlled motion experiment
│   ├── evaluation/      metrics, colour rendering, videos, colour-chart reflectance
│   ├── config.py        default hyperparameters
│   ├── paths.py         data and output locations (environment variables INSF_*)
│   └── cli.py           command table used by main.py
└── data/README.md       files the code expects (no data are distributed)
```

`python main.py` lists all commands; `python main.py <group> <command> --help` lists the arguments.
Values in a `--config` file are passed as command-line arguments; arguments given after it override them.

## Installation

Requires an NVIDIA GPU with CUDA, PyTorch with CUDA and the tiny-cuda-nn PyTorch bindings.

```bash
conda create -n insf python=3.12 && conda activate insf
pip install torch --index-url https://download.pytorch.org/whl/cu128          # match your CUDA
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
pip install -e .
```

The paper used PyTorch 2.11 and CUDA 13.0 on an NVIDIA RTX PRO 6000 Blackwell GPU.

## Data

This repository contains code only.  [`data/README.md`](data/README.md) lists the files that the
commands expect; requests for the data should be addressed to the corresponding author.

```bash
python main.py data make-mask
python main.py data convert <raw folder> data/measurements/move2
python main.py data make-holdout
```

## Experiments

Run from the repository root; outputs are written to `outputs/` unless `INSF_OUT` is set.

| Experiment | Command |
|---|---|
| Static scene, shared field (move1) | `python main.py reconstruct shared-field --config configs/move1_shared.yaml` |
| Mean-input single-frame INR (move1) | `python main.py reconstruct mean-input --config configs/move1_mean_input.yaml` |
| Single-frame INR on all move2 frames | `python main.py reconstruct single-frame --config configs/move2_single_frame.yaml` |
| Optical flow from the single-frame cubes | `python main.py reconstruct flow --config configs/move2_flow.yaml` |
| Single-frame INR with held-out pixels | `python main.py reconstruct mc-window --config configs/move2_single_frame_holdout.yaml` |
| Motion-compensated multiframe INR | `python main.py reconstruct mc-window --config configs/move2_mc_window.yaml` |
| TwIST / ADMM on move2 | `python main.py baseline twist --config configs/baselines_twist.yaml` (same for `admm`) |
| MST / MST++ on move2 | `python main.py baseline mst --config configs/baselines_mst.yaml --train-until <unix time>` |
| Public datasets (KAIST, CAVE, Harvard) | `python main.py simulate generalization-prep`, then `python main.py simulate generalization` (one per GPU), `python main.py evaluate generalization` |
| MST without the Harvard test scenes | `python main.py baseline mst --arch mst --exclude-harvard-holdout --out outputs/generalization/runs/mst_holdout10 --train-until <unix time>` |
| Controlled motion experiment | `python main.py simulate motion --config configs/motion_simulation.yaml` |
| Colour chart (real captures) | `python main.py data prepare-colorchecker`, `python main.py reconstruct colorchecker --config configs/colorchecker.yaml`, `python main.py evaluate colorchecker outputs/colorchecker/runs` |

The motion-compensated run in the paper was split into three chunks of the 98 frames
(`--start 0 --end 33`, `--start 33 --end 66`, `--start 66 --end 98`), one per GPU.

## Third-party code and references

| Code | Used for | Source |
|---|---|---|
| tiny-cuda-nn (NVIDIA) | multiresolution hash encoding and fully fused MLP of the field | https://github.com/NVlabs/tiny-cuda-nn |
| MST (Cai et al., CVPR 2022), MIT License | MST baseline | https://github.com/caiyuanhao1998/MST |
| MST++ (Cai et al., CVPRW 2022), MIT License | MST++ baseline | https://github.com/caiyuanhao1998/MST-plus-plus |
| Space–time neural field code of *Video-rate gigapixel ptychography via space-time neural field representations* | reference for the hash-grid neural field and its training loop | code released with that paper |
| OpenCV | DIS optical flow between reconstructions | https://opencv.org |
| colour-science | ColorChecker reference reflectances (Ohta) | https://www.colour-science.org |

TwIST (Bioucas-Dias and Figueiredo, 2007) and ADMM (Boyd et al., 2011) are implemented in
`insf/baselines/` from the published algorithms.

MST and MST++ are not included.  Clone the two official repositories next to this repository
(or set `INSF_THIRD_PARTY` to the folder that contains them):

```bash
git clone https://github.com/caiyuanhao1998/MST ../MST
git clone https://github.com/caiyuanhao1998/MST-plus-plus ../MST-plus-plus
```

`insf/baselines/mst.py` loads their architecture files without modifying them and adapts them at
import time to this camera: the dispersion shift of CASSI is replaced by the identity, the input has
124 channels (masked measurement, validity, minimum-norm estimate and filter responses on 61 spectral
knots) and the output has 61 knots (350:10:950 nm) that are interpolated to the 601 bands.

## Citation

Please cite the paper if you use this code (reference to be added after publication).

## License

MIT License, see [LICENSE](LICENSE).
