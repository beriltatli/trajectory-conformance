import numpy as np

_POINT_BLOCK = 4096


def cross_track_distance(east: np.ndarray, north: np.ndarray, path_en: np.ndarray) -> np.ndarray:
    """Distance in metres from each (east, north) point to the nearest segment of a polyline.

    path_en is (M, 2) in the same local tangent frame as the points. Points are processed in
    blocks so a long flight against a long path stays within a few MB.
    """
    points = np.column_stack([east, north]).astype(float)
    a = path_en[:-1]
    b = path_en[1:]
    ab = b - a
    length2 = np.einsum("ij,ij->i", ab, ab)
    out = np.empty(len(points))
    for lo in range(0, len(points), _POINT_BLOCK):
        p = points[lo:lo + _POINT_BLOCK, None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.einsum("ijk,jk->ij", p - a, ab) / length2
        u = np.clip(np.nan_to_num(u), 0.0, 1.0)
        nearest = a + u[..., None] * ab
        out[lo:lo + _POINT_BLOCK] = np.linalg.norm(p - nearest, axis=-1).min(axis=1)
    return out


def exceedances(east: np.ndarray, north: np.ndarray, path_en: np.ndarray, threshold_m: float) -> np.ndarray:
    """No filter, no model: alert whenever the reported position is too far from the path."""
    return cross_track_distance(east, north, path_en) > threshold_m
