import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    import tinycudann as tcnn
except (ImportError, OSError) as error:
    raise ImportError(
        "Cannot load tiny-cuda-nn. Install PyTorch bindings that match "
        "the current CUDA and GPU architecture."
    ) from error


class SpatialSpectralField(nn.Module):

    def __init__(
        self,
        *,
        num_spectral_knots: int,
        output_scale: float,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 18,
        base_resolution: int = 16,
        per_level_scale: float = 1.32,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_spectral_knots < 2:
            raise ValueError("num_spectral_knots must be at least 2.")
        if hidden_dim not in (16, 32, 64, 128):
            raise ValueError("hidden_dim must be 16, 32, 64, or 128.")

        self.feature_dim = n_levels * n_features_per_level
        self.num_spectral_knots = num_spectral_knots

        self.spatial_encoder = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.renderer = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": hidden_layers,
            },
        )


        self.spectral_gamma_knots = nn.Parameter(
            torch.empty(num_spectral_knots, self.feature_dim)
        )
        self.spectral_beta_knots = nn.Parameter(
            torch.empty(num_spectral_knots, self.feature_dim)
        )
        nn.init.normal_(self.spectral_gamma_knots, mean=1.0, std=0.01)
        nn.init.normal_(self.spectral_beta_knots, mean=0.0, std=0.01)
        self.register_buffer(
            "output_scale",
            torch.tensor(float(output_scale), dtype=torch.float32),
        )

    def interpolate_film_parameters(
        self,
        wavelengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = wavelengths.reshape(-1).clamp(0.0, 1.0) * (
            self.num_spectral_knots - 1
        )
        lower = positions.floor().long()
        upper = (lower + 1).clamp(max=self.num_spectral_knots - 1)
        weights = (positions - lower).unsqueeze(-1)
        gamma = torch.lerp(
            self.spectral_gamma_knots[lower],
            self.spectral_gamma_knots[upper],
            weights,
        )
        beta = torch.lerp(
            self.spectral_beta_knots[lower],
            self.spectral_beta_knots[upper],
            weights,
        )
        return gamma, beta

    def forward(
        self,
        xy: torch.Tensor,
        wavelengths: torch.Tensor,
    ) -> torch.Tensor:
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must have shape [B,2], got {tuple(xy.shape)}.")

        spatial = self.spatial_encoder(xy.contiguous())
        gamma, beta = self.interpolate_film_parameters(wavelengths)
        fused = spatial[:, None, :] * gamma[None, :, :] + beta[None, :, :]

        batch_size, num_bands = fused.shape[:2]
        raw = self.renderer(fused.reshape(-1, self.feature_dim).contiguous()).reshape(
            batch_size, num_bands
        )


        return F.softplus(raw.float()) * (self.output_scale / math.log(2.0))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class SpatioTemporalField(SpatialSpectralField):

    def __init__(self, *, log2_hashmap_size_3d: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spatial_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": kwargs.get("n_levels", 16),
                "n_features_per_level": kwargs.get("n_features_per_level", 2),
                "log2_hashmap_size": log2_hashmap_size_3d,
                "base_resolution": kwargs.get("base_resolution", 16),
                "per_level_scale": kwargs.get("per_level_scale", 1.32),
            },
        )

    def forward(self, xyt: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        feat = self.spatial_encoder(xyt.contiguous())
        gamma, beta = self.interpolate_film_parameters(wavelengths)
        fused = feat[:, None, :] * gamma[None] + beta[None]
        b, l = fused.shape[:2]
        raw = self.renderer(fused.reshape(-1, self.feature_dim).contiguous()).reshape(b, l)
        return F.softplus(raw.float()) * (self.output_scale / math.log(2.0))
