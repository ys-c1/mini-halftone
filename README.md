# mini-halftone

A tiny, dependency-free Python 3 halftone/dithering library. It works on
rectangular 8-bit grayscale images represented as `list[list[int]]` with
samples in `0..255`.

The current checkout is the **exercise (buggy) build**: three defects are
intentionally planted in `src/halftone.py`, so `make test` is expected to
report assertion failures. The corrections are deliberately not listed here.

## Build and test

```sh
make test
# or
./run_tests.sh
# or directly
python3 tests/test_halftone.py
```

The tests need only the Python 3 standard library. The runner prints one line
per check and finishes with `checks: N, failures: M`; it exits non-zero when
`M > 0`.

## API

```python
floyd_steinberg(gray: list[list[int]], levels: int = 2, serpentine: bool = True) -> list[list[int]]
ordered_dither(gray, levels=2, matrix_size: int = 4) -> list[list[int]]
mean_abs_error(a, b) -> float
```

- `levels` is the number of quantization levels. Levels are evenly spaced and
  include both endpoints, so `levels=2` uses `{0, 255}` and `levels=4` uses
  `{0, 85, 170, 255}`.
- `serpentine=True` scans odd rows right-to-left and mirrors the
  Floyd-Steinberg kernel so error diffusion follows the scan direction.
- `ordered_dither` uses a recursively generated Bayer matrix. `matrix_size`
  must be a positive power of two.
- `mean_abs_error` compares non-overlapping 4x4 tile means and returns their
  mean absolute difference. Full-resolution per-pixel MAE is not a useful
  halftone metric: a perfect 50% gray halftone would score about 127 because
  every pixel is either 0 or 255.
- Empty images return `[]`. Out-of-range input samples are accepted, and every
  output sample is clamped to `0..255`.

## Layout

```text
src/halftone.py          implementation
tests/test_halftone.py   assertion-based test runner
Makefile                 make test / make clean
run_tests.sh             shell wrapper around the test runner
```
