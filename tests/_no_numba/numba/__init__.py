"""Stands in for numba when a test needs a Python without it (tests/test_kernels.py)."""

raise ImportError("numba is blocked for this test")
