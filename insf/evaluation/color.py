from pathlib import Path

import numpy as np

WAVELENGTHS_NM = np.arange(350, 951)


def cie_matrix(xyzbar_path: Path) -> np.ndarray:
    from scipy.io import loadmat

    table = np.asarray(loadmat(xyzbar_path)["xyzbar"], dtype=np.float64)
    grid = np.arange(400, 721, 10)
    cmf = np.zeros((WAVELENGTHS_NM.size, 3), dtype=np.float64)
    inside = (WAVELENGTHS_NM >= 400) & (WAVELENGTHS_NM <= 720)
    for c in range(3):
        cmf[inside, c] = np.interp(WAVELENGTHS_NM[inside], grid, table[:, c])
    xyz2srgb = np.array(
        [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0414], [0.0557, -0.2040, 1.0570]]
    )
    return (cmf @ xyz2srgb.T).astype(np.float32)
