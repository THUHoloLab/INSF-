from dataclasses import asdict, dataclass
from pathlib import Path


from insf import paths

CODE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:


    mask_path: Path = paths.MASK
    measurement_path: Path = paths.STATIC / "2025-08-15--10-55-46-597.mat"
    mask_key: str = "dataCube"
    measurement_key: str = "measurement"
    crop_size: int = 800
    original_roi_size: int = 1000
    original_roi_start_matlab: tuple[int, int] = (200, 200)
    measurement_gain: float = 0.5


    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 18
    base_resolution: int = 16
    per_level_scale: float = 1.32


    num_spectral_knots: int = 48
    hidden_dim: int = 64
    hidden_layers: int = 2


    epochs: int = 100
    batch_size: int = 1024
    steps_per_epoch: int | None = None
    learning_rate: float = 1e-3
    min_learning_rate: float = 5e-5
    spectral_tv_weight: float = 2e-2

    spatial_tv_weight: float = 5e-2
    huber_beta: float = 2e-2
    tv_epsilon: float = 1e-3
    seed: int = 2026


    preview_band: int = 150
    save_every: int = 10
    reconstruction_pixel_batch: int = 8192
    reconstruction_spectral_batch: int = 32
    compress_mat: bool = True
    output_root: Path = paths.OUT / "single_frame"

    def to_dict(self) -> dict:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
