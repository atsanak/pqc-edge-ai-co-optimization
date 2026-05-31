"""
Preload CUDA shared libraries that PyTorch needs but may not find
when LD_LIBRARY_PATH is not set (e.g. venv activated before the
library-path patch was added, or running from an IDE/cron).

Usage — call **before** ``import torch``::

    from cuda_preload import preload_cuda_libs
    preload_cuda_libs()

The function is safe to call on any platform; it silently does nothing
if the libraries are not present (macOS, CPU-only installs, etc.).
"""

import ctypes
import glob
import os
import sys


def preload_cuda_libs() -> None:
    """Force-load CUDA shared objects into the process so ``import torch`` succeeds.

    On Jetson (and pip-installed ``nvidia-cusparselt-cu12``), the
    ``libcusparseLt.so.0`` lives inside the pip package tree and is not
    on the default linker search path.  Loading it with ``ctypes.CDLL``
    makes the library available before the torch C-extension calls
    ``dlopen``.

    NOTE: We intentionally use ``mode=0`` (``RTLD_LOCAL``) instead of
    ``RTLD_GLOBAL``.  ``RTLD_GLOBAL`` leaks cuSPARSELt symbols into the
    global namespace and causes a "free(): invalid pointer" crash when
    OpenCV's Qt5 backend is loaded.  ``RTLD_LOCAL`` avoids the conflict
    while still allowing PyTorch to resolve the library at runtime.
    """
    # cuSPARSELt — the most common missing lib on Jetson with pip torch
    _try_preload_glob(
        os.path.join(
            sys.prefix,
            "lib", "python*", "site-packages",
            "nvidia", "cusparselt", "lib", "libcusparseLt.so*",
        )
    )

    # Also try the CUDA toolkit system path (in case the pip package is absent)
    _try_preload_glob("/usr/local/cuda*/lib64/libcusparseLt.so*")


def _try_preload_glob(pattern: str) -> bool:
    """Load the first matching .so and return True on success."""
    for path in sorted(glob.glob(pattern)):
        try:
            ctypes.CDLL(path, mode=0)  # RTLD_LOCAL — avoids Qt5/OpenCV crash
            return True
        except OSError:
            continue
    return False
