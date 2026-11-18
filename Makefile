PYTHON ?= python3

.PHONY: all test clean

all: test

test:
	$(PYTHON) tests/test_halftone.py

clean:
	rm -rf build __pycache__ src/__pycache__ tests/__pycache__
