#!/usr/bin/env python3
"""Assertion-based tests for the mini-halftone module.

Run directly with ``python3 tests/test_halftone.py`` or through ``make test``.
Every check is counted; a failed check is reported as an assertion failure and
the runner continues, so the total failure count is stable and the process
never dies on the first defect.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import halftone  # noqa: E402

CHECKS = 0
FAILURES = 0


def check(condition: bool, label: str) -> None:
    global CHECKS, FAILURES
    CHECKS += 1
    try:
        assert condition, label
    except AssertionError:
        FAILURES += 1
        print("FAIL [%02d] %s" % (CHECKS, label))
    else:
        print("PASS [%02d] %s" % (CHECKS, label))


def flat(image):
    return [value for row in image for value in row]


def all_in(image, allowed):
    return all(value in allowed for value in flat(image))


def const_image(height, width, value):
    return [[value for _ in range(width)] for _ in range(height)]


def diagonal_ramp(height, width):
    span = (height - 1) + (width - 1)
    return [[(x + y) * 255 // span for x in range(width)] for y in range(height)]


def horizontal_ramp(height, width):
    return [[x * 255 // (width - 1) for x in range(width)] for _ in range(height)]


def test_floyd_steinberg_gray_128():
    image = const_image(16, 16, 128)
    out = halftone.floyd_steinberg(image, levels=2, serpentine=True)

    check(
        len(out) == 16 and all(len(row) == 16 for row in out),
        "Floyd-Steinberg preserves a 16x16 shape",
    )
    check(
        all_in(out, {0, 255}),
        "Floyd-Steinberg on gray 128 emits only 0/255 (got %r)"
        % sorted(set(flat(out))),
    )
    check(
        0 in set(flat(out)) and 255 in set(flat(out)),
        "Floyd-Steinberg on gray 128 mixes black and white",
    )
    error = halftone.mean_abs_error(image, out)
    check(
        error < 6.0,
        "Floyd-Steinberg gray-128 block mean error %.3f < 6" % error,
    )


def test_serpentine_scan():
    image = diagonal_ramp(16, 16)
    serpentine = halftone.floyd_steinberg(image, levels=2, serpentine=True)
    left_to_right = halftone.floyd_steinberg(image, levels=2, serpentine=False)

    check(
        serpentine != left_to_right,
        "serpentine=True and serpentine=False produce different images",
    )
    error_serpentine = halftone.mean_abs_error(image, serpentine)
    error_left_to_right = halftone.mean_abs_error(image, left_to_right)
    check(
        error_serpentine < 10.0,
        "serpentine block mean error %.3f < 10" % error_serpentine,
    )
    check(
        error_left_to_right < 10.0,
        "non-serpentine block mean error %.3f < 10" % error_left_to_right,
    )
    check(
        all_in(serpentine, {0, 255}) and all_in(left_to_right, {0, 255}),
        "both serpentine modes emit only 0/255 at levels=2",
    )


def test_ordered_dither_4x4():
    image = const_image(16, 16, 128)
    out = halftone.ordered_dither(image, levels=2, matrix_size=4)
    values = set(flat(out))

    check(
        all_in(out, {0, 255}),
        "4x4 Bayer on gray 128 emits only 0/255 (got %r)" % sorted(values),
    )
    check(
        0 in values and 255 in values,
        "4x4 Bayer on gray 128 mixes black and white",
    )
    error = halftone.mean_abs_error(image, out)
    check(error < 12.0, "4x4 Bayer block mean error %.3f < 12" % error)


def test_four_levels():
    image = horizontal_ramp(16, 16)
    out = halftone.floyd_steinberg(image, levels=4, serpentine=True)
    values = set(flat(out))
    allowed = {0, 85, 170, 255}

    check(
        all_in(out, allowed),
        "levels=4 emits only the four allowed gray values (got %r)"
        % sorted(values),
    )
    check(
        len(values) == 4,
        "levels=4 uses exactly four distinct gray values (got %d)"
        % len(values),
    )


def test_empty_images():
    check(
        halftone.floyd_steinberg([], levels=2) == [],
        "Floyd-Steinberg on an empty image returns []",
    )
    check(
        halftone.ordered_dither([], levels=2, matrix_size=4) == [],
        "ordered dither on an empty image returns []",
    )


def test_out_of_range_samples_are_clamped():
    bright = const_image(8, 8, 300)
    dark = const_image(8, 8, -50)
    bright_out = halftone.floyd_steinberg(bright, levels=2)
    dark_out = halftone.floyd_steinberg(dark, levels=2)

    check(
        all(value == 255 for value in flat(bright_out)),
        "uniform 300 clamps to 255 (got %r)" % sorted(set(flat(bright_out))),
    )
    check(
        all(value == 0 for value in flat(dark_out)),
        "uniform -50 clamps to 0 (got %r)" % sorted(set(flat(dark_out))),
    )

    mixed = [
        [-50, 300, -200, 400],
        [300, -50, 400, -200],
        [-200, 400, -50, 300],
        [400, -200, 300, -50],
    ]
    for levels in (2, 4):
        out = halftone.floyd_steinberg(mixed, levels=levels)
        allowed = {round(i * 255 / (levels - 1)) for i in range(levels)}
        check(
            all(0 <= value <= 255 for value in flat(out)),
            "levels=%d clamps every output into 0..255 (got %r)"
            % (levels, sorted(set(flat(out)))),
        )
        check(
            all_in(out, allowed),
            "levels=%d clamps every output to an allowed level (got %r)"
            % (levels, sorted(set(flat(out)))),
        )


def main():
    test_floyd_steinberg_gray_128()
    test_serpentine_scan()
    test_ordered_dither_4x4()
    test_four_levels()
    test_empty_images()
    test_out_of_range_samples_are_clamped()

    print("checks: %d, failures: %d" % (CHECKS, FAILURES))
    if FAILURES == 0:
        print("ALL TESTS PASSED")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
