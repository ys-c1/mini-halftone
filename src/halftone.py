"""Dependency-free halftoning and dithering for 8-bit grayscale images.

The public API works on rectangular ``list[list[int]]`` images whose samples
are in ``0..255``.  Out-of-range input samples are accepted and the produced
output is always clamped back to ``0..255``.

``mean_abs_error`` is a low-frequency fidelity metric: both images are reduced
to non-overlapping 4x4 tile means and the mean absolute difference of those
tile means is returned.  Full-resolution per-pixel MAE is not meaningful for a
binary halftone (a 50% gray field would score about 127 even when perfect), so
the tile means are used instead.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

BLOCK_SIZE = 4

__all__ = ["floyd_steinberg", "ordered_dither", "mean_abs_error"]


def _clamp255(value: float) -> int:
    """Clamp a numeric value to an integer in the inclusive 0..255 range."""

    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return int(round(value))


def _validate_levels(levels: int) -> None:
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
        raise ValueError("levels must be an integer >= 2")


def _image_shape(gray: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Return (height, width) and reject ragged or non-list input."""

    if not isinstance(gray, (list, tuple)):
        raise TypeError("image must be a list of rows")
    if len(gray) == 0:
        return 0, 0
    if not isinstance(gray[0], (list, tuple)):
        raise TypeError("each row must be a list of integers")
    width = len(gray[0])
    for row in gray:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError("image rows must all have the same length")
    return len(gray), width


def _quantize(value: float, levels: int) -> int:
    """Quantize one sample to the nearest of ``levels`` evenly spaced levels.

    The returned sample is clamped back to 0..255.  Without the final clamp,
    out-of-range samples can leave the output outside the valid gray range.
    """

    step = 255.0 / (levels - 1)
    index = int(value / step + 0.5)
    quantized = index * step
    return int(round(quantized))


def floyd_steinberg(
    gray: List[List[int]], levels: int = 2, serpentine: bool = True
) -> List[List[int]]:
    """Quantize ``gray`` with Floyd-Steinberg error diffusion.

    ``levels`` is the number of quantization levels (2 gives 0/255, 4 gives
    0/85/170/255, and so on).  With ``serpentine=True`` odd rows are scanned
    right-to-left and the diffusion kernel is mirrored so errors keep moving
    in the scan direction.
    """

    _validate_levels(levels)
    if not isinstance(serpentine, bool):
        raise ValueError("serpentine must be a bool")

    height, width = _image_shape(gray)
    if height == 0 or width == 0:
        if height == 0:
            return []
        return [[] for _ in range(height)]

    # Work in floating point so error diffusion is not rounded at every step.
    work = [[float(gray[y][x]) for x in range(width)] for y in range(height)]
    out = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        left_to_right = (not serpentine) or (y % 2 == 0)
        columns = range(width) if left_to_right else range(width - 1, -1, -1)
        for x in columns:
            old_value = work[y][x]
            new_value = _quantize(old_value, levels)
            out[y][x] = new_value

            error = old_value - new_value
            step_x = 1
            # (dy, dx, weight): 7/16 is the next sample in scan direction,
            # 3/16 and 1/16 mirror across the next row, 5/16 is straight down.
            for dy, dx, weight in (
                (0, step_x, 7),
                (1, -step_x, 3),
                (1, 0, 5),
                (1, step_x, 1),
            ):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    work[ny][nx] += error * weight / 8.0

    return out


def _bayer_matrix(size: int) -> List[List[int]]:
    """Build a recursively defined Bayer threshold matrix of power-of-two size."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("matrix_size must be a positive power of two")
    if size & (size - 1):
        raise ValueError("matrix_size must be a positive power of two")

    matrix = [[0]]
    current = 1
    while current < size:
        nxt = current * 2
        expanded = [[0 for _ in range(nxt)] for _ in range(nxt)]
        for y in range(current):
            for x in range(current):
                base = matrix[y][x] * 4
                expanded[y][x] = base
                expanded[y][x + current] = base + 2
                expanded[y + current][x] = base + 3
                expanded[y + current][x + current] = base + 1
        matrix = expanded
        current = nxt
    return matrix


def ordered_dither(
    gray: List[List[int]], levels: int = 2, matrix_size: int = 4
) -> List[List[int]]:
    """Quantize ``gray`` with a Bayer ordered-dither threshold matrix."""

    _validate_levels(levels)
    matrix = _bayer_matrix(matrix_size)
    height, width = _image_shape(gray)
    if height == 0 or width == 0:
        if height == 0:
            return []
        return [[] for _ in range(height)]

    step = 255.0 / (levels - 1)
    cells = matrix_size * matrix_size
    out = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            threshold = (matrix[y % matrix_size][x % matrix_size] + 0.5) / cells
            dithered = gray[y][x] + (threshold - 0.5) * step
            out[y][x] = _quantize(dithered, levels)

    return out


def mean_abs_error(
    a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]
) -> float:
    """Mean absolute difference of 4x4 tile means of two same-sized images.

    For an empty image the error is defined as ``0.0``.  Ragged or mismatched
    images raise ``ValueError``.
    """

    height_a, width_a = _image_shape(a)
    height_b, width_b = _image_shape(b)
    if (height_a, width_a) != (height_b, width_b):
        raise ValueError("images must have the same shape")
    if height_a == 0 or width_a == 0:
        return 0.0

    total = 0.0
    tiles = 0
    for top in range(0, height_a, BLOCK_SIZE):
        for left in range(0, width_a, BLOCK_SIZE):
            sum_a = 0
            sum_b = 0
            count = 0
            bottom = min(top + BLOCK_SIZE, height_a)
            right = min(left + BLOCK_SIZE, width_a)
            for y in range(top, bottom):
                for x in range(left, right):
                    sum_a += a[y][x]
                    sum_b += b[y][x]
                    count += 1
            total += abs(sum_a / count - sum_b / count)
            tiles += 1

    return total / tiles
