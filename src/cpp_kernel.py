"""
C++17 Quantitative Kernel Bridge (`src/cpp_kernel.py`).

Automatically compiles `src/cpp/quant_core.cpp` into a native shared library
using Apple clang++ / g++ (`-O3 -std=c++17 -shared -fPIC`) and binds it via `ctypes`.
Provides transparent pure-Python fallbacks if a C++ compiler is unavailable.
"""

from __future__ import annotations

import ctypes
import math
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

CPP_DIR = Path(__file__).resolve().parent / "cpp"
CPP_SOURCE = CPP_DIR / "quant_core.cpp"
LIB_EXT = ".dylib" if platform.system() == "Darwin" else ".so"
LIB_PATH = CPP_DIR / f"libquant_core{LIB_EXT}"

_LIB_INSTANCE: Optional[ctypes.CDLL] = None
_COMPILE_ATTEMPTED: bool = False


def _compile_shared_library() -> Optional[Path]:
    """Compile `quant_core.cpp` into a native shared library if needed."""
    if not CPP_SOURCE.exists():
        return None

    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= CPP_SOURCE.stat().st_mtime:
        return LIB_PATH

    compilers = ["/usr/bin/clang++", "clang++", "/usr/bin/g++", "g++"]
    for compiler in compilers:
        cmd = [
            compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(CPP_SOURCE),
            "-o",
            str(LIB_PATH),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
            if LIB_PATH.exists():
                return LIB_PATH
        except Exception:
            continue
    return None


def get_kernel() -> Optional[ctypes.CDLL]:
    """Load and configure the C++17 shared library, compiling on first use."""
    global _LIB_INSTANCE, _COMPILE_ATTEMPTED
    if _LIB_INSTANCE is not None:
        return _LIB_INSTANCE
    if _COMPILE_ATTEMPTED:
        return None

    _COMPILE_ATTEMPTED = True
    lib_file = _compile_shared_library()
    if lib_file is None:
        return None

    try:
        lib = ctypes.CDLL(str(lib_file))

        # 1. cpp_bsm_price
        lib.cpp_bsm_price.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.cpp_bsm_price.restype = ctypes.c_double

        # 2. cpp_bsm_greeks
        lib.cpp_bsm_greeks.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.cpp_bsm_greeks.restype = None

        # 3. cpp_bsm_implied_vol
        lib.cpp_bsm_implied_vol.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.cpp_bsm_implied_vol.restype = ctypes.c_double

        # 4. cpp_bsm_mc_revaluation
        lib.cpp_bsm_mc_revaluation.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.cpp_bsm_mc_revaluation.restype = None

        # 5. cpp_ewma_volatility
        lib.cpp_ewma_volatility.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.cpp_ewma_volatility.restype = None

        _LIB_INSTANCE = lib
        return _LIB_INSTANCE
    except Exception:
        return None


def bsm_price_fast(
    S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool = True
) -> float:
    """Compute BSM European option price via C++17 kernel (with Python fallback)."""
    lib = get_kernel()
    if lib is not None:
        return float(
            lib.cpp_bsm_price(
                float(S),
                float(K),
                float(r),
                float(q),
                float(sigma),
                float(T),
                1 if is_call else 0,
            )
        )
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if is_call:
        return float(S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2))
    return float(K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1))


def bsm_greeks_fast(
    S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool = True
) -> Tuple[float, float, float, float, float]:
    """Return (price, delta, gamma, vega, theta) via C++17 kernel."""
    lib = get_kernel()
    if lib is not None:
        p = ctypes.c_double()
        d = ctypes.c_double()
        g = ctypes.c_double()
        v = ctypes.c_double()
        th = ctypes.c_double()
        lib.cpp_bsm_greeks(
            float(S),
            float(K),
            float(r),
            float(q),
            float(sigma),
            float(T),
            1 if is_call else 0,
            ctypes.byref(p),
            ctypes.byref(d),
            ctypes.byref(g),
            ctypes.byref(v),
            ctypes.byref(th),
        )
        return float(p.value), float(d.value), float(g.value), float(v.value), float(th.value)

    # Pure Python fallback
    if T <= 1e-12 or sigma <= 1e-12:
        price = bsm_price_fast(S, K, r, q, sigma, T, is_call)
        delta = (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
        return price, delta, 0.0, 0.0, 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    pdf_d1 = float(norm.pdf(d1))
    price = bsm_price_fast(S, K, r, q, sigma, T, is_call)
    gamma = (df_q * pdf_d1) / (S * sigma * sqrt_T)
    vega = S * df_q * pdf_d1 * sqrt_T
    if is_call:
        delta = float(df_q * norm.cdf(d1))
        theta = float(
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * df_r * norm.cdf(d2)
            + q * S * df_q * norm.cdf(d1)
        )
    else:
        delta = float(-df_q * norm.cdf(-d1))
        theta = float(
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * df_r * norm.cdf(-d2)
            - q * S * df_q * norm.cdf(-d1)
        )
    return price, delta, gamma, vega, theta


def ewma_volatility_fast(
    returns: np.ndarray, lambda_decay: float = 0.94, init_var: Optional[float] = None
) -> np.ndarray:
    """Compute RiskMetrics EWMA daily volatility array via C++17 kernel."""
    arr = np.ascontiguousarray(returns, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.array([], dtype=np.float64)
    v0 = float(arr[0] ** 2 if init_var is None else init_var)
    out = np.empty(n, dtype=np.float64)

    lib = get_kernel()
    if lib is not None:
        lib.cpp_ewma_volatility(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_double(lambda_decay),
            ctypes.c_double(v0),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out

    var_t = v0
    out[0] = math.sqrt(var_t)
    for t in range(1, n):
        var_t = lambda_decay * var_t + (1.0 - lambda_decay) * (arr[t - 1] ** 2)
        out[t] = math.sqrt(var_t)
    return out
