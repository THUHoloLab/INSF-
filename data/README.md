# Data

This repository contains code only.  No measurements, coded masks, filter responses or other
camera data are distributed.  The scripts expect the following files; their locations can be
changed with the environment variables in `insf/paths.py`.

| Expected path | Contents |
|---|---|
| `data/mask/mask_3D.mat` | coded response of the sensor, 1200 × 1600 × 601 (350:1:950 nm), HDF5 / MATLAB v7.3, dataset `dataCube` |
| `data/mask/qe.npy` | per-channel response, [601, 10] (wavelength + 9 channels); only for the colour-chart commands |
| `data/measurements/<sequence>/*.mat` | coded frames, key `measurement`, 1200 × 1600 |
| `data/holdout_10pct_seed20261019.npy` | fixed 10 % test pixels of move2; regenerate with `python main.py data make-holdout` |
| `data/xyzbar.mat` | CIE 1931 2° colour-matching functions, key `xyzbar`, 400:10:720 nm (colour images only; available from cvrl.org) |
| `data/colorchecker_raw/<scene>/*.raw` | raw colour-chart captures |
| `data/harvard_holdout.json` | `{"held_out": ["imgXX.mat", ...]}`: Harvard scenes excluded from MST training and used as test scenes (`simulate generalization`) |
| `data/colorchecker_frames.json` | `{"<key>": {"scene": <folder>, "idx": <frame index>, "roi": [row, col], "corners": [[x, y] x 4]}}`: selected colour-chart frames, 640 × 640 region and chart corners in super-pixel units (`reconstruct colorchecker`) |
| `data/public/{harvard,cave,kaist}` | public hyperspectral datasets (download from the dataset websites) |

`python main.py data make-mask` builds `mask_3D.mat` and `qe.npy` from a QE sheet of the 9 filter channels
(`data/mask/filter_qe.xlsx`) and a channel layout file (`data/mask/channel_layout.json`,
`{"pattern": [[..], [..], [..]]}`, channel numbers 1–9 on the 3 × 3 super-pixel).
`python main.py data convert` converts raw camera frames to the `.mat` format.

Requests for the data should be addressed to the corresponding author.
