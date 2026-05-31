"""
GPU selection utility — no CUDA dependencies.

Import and call ``select_gpu()`` **before** importing any CUDA-aware library
(PyTorch, TensorFlow, Ultralytics) so that ``CUDA_VISIBLE_DEVICES`` is visible
to the runtime from the start.

Cross-platform support:
  - NVIDIA desktop/server GPUs  (nvidia-smi with full query support)
  - NVIDIA Jetson / Tegra        (nvidia-smi with partial query + /etc/nv_tegra_release)
  - Apple Silicon MPS            (macOS Metal)
  - CPU fallback
"""

import os
import platform
import subprocess


# ── Jetson / Tegra helpers ──────────────────────────────────────────

def is_jetson() -> bool:
    """Detect NVIDIA Jetson / Tegra platform."""
    # Method 1: /etc/nv_tegra_release exists on all Jetson Linux (L4T) installs
    if os.path.isfile("/etc/nv_tegra_release"):
        return True
    # Method 2: device-tree model string contains "Jetson"
    try:
        with open("/proc/device-tree/model", "r") as f:
            if "jetson" in f.read().lower():
                return True
    except Exception:
        pass
    return False


def _get_jetson_total_ram_mb() -> float:
    """Return total system RAM in MB (Jetson uses unified CPU/GPU memory)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0  # kB → MB
    except Exception:
        pass
    return 0.0


def _get_jetson_gpu_info() -> list:
    """
    Return GPU list for a Jetson board.

    nvidia-smi *is* present on modern JetPack (≥ 6) but the CSV query returns
    ``[N/A]`` for memory.  We parse what we can from nvidia-smi and fill in
    shared-memory info from /proc/meminfo (Jetson uses unified CPU/GPU RAM).
    """
    gpus = []
    name = "Jetson GPU"

    # Try nvidia-smi for the GPU name
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        for line in out.strip().splitlines():
            parts = [s.strip() for s in line.split(",", 1)]
            if len(parts) == 2:
                idx = int(parts[0])
                name = parts[1]
    except Exception:
        pass

    total_mem_mb = _get_jetson_total_ram_mb()
    gpus.append((0, name, total_mem_mb))
    return gpus


# ── nvidia-smi desktop/server parser ───────────────────────────────

def _get_nvidia_smi_gpus() -> list:
    """
    Probe GPUs via ``nvidia-smi`` CSV query.
    Gracefully handles ``[N/A]`` fields (common on Jetson / Tegra).
    Returns list of (index, name, memory_MB).
    """
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        for line in out.strip().splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) < 3:
                continue
            idx_s, name = parts[0], parts[1]
            mem_s = parts[2]
            try:
                idx = int(idx_s)
            except ValueError:
                continue
            # memory.total may be "[N/A]" on Jetson
            try:
                mem = float(mem_s)
            except (ValueError, TypeError):
                mem = 0.0  # will be patched by Jetson helper if applicable
            gpus.append((idx, name, mem))
    except Exception:
        pass
    return gpus


# ── Main entry point ───────────────────────────────────────────────

def select_gpu() -> int:
    """
    Interactive GPU selector.

    Probes GPUs via ``nvidia-smi`` (no CUDA import needed), lets the user pick
    one, and sets ``CUDA_VISIBLE_DEVICES`` so the chosen card appears as
    device 0 to every framework.

    Returns the **physical** GPU index that was selected (useful for pynvml
    monitoring, which is unaffected by ``CUDA_VISIBLE_DEVICES``).
    """
    print("\n" + "=" * 60)
    print("          GPU / Device Selection")
    print("=" * 60)

    _is_jetson = is_jetson()

    # --- Detect GPUs via nvidia-smi ---
    gpus = _get_nvidia_smi_gpus()

    # --- Jetson fallback: nvidia-smi may return empty or 0-MB memory ---
    if _is_jetson and not gpus:
        gpus = _get_jetson_gpu_info()
    elif _is_jetson and gpus:
        # Fix up missing memory values using system RAM (unified memory)
        fixed = []
        for idx, name, mem in gpus:
            if mem == 0.0:
                mem = _get_jetson_total_ram_mb()
            fixed.append((idx, name, mem))
        gpus = fixed

    # --- MPS (macOS Apple Silicon) ---
    if not gpus and platform.system() == "Darwin":
        print(f"\nFound Apple MPS GPU (Metal Performance Shaders)")
        print(f"  [0]  {platform.processor() or 'Apple Silicon'} — MPS")
        print(f"  [c]  CPU")
        choice = input("\nUse MPS? [Y/n/c]: ").strip().lower()
        if choice in ("", "y", "yes", "0"):
            print("\n>> Using MPS\n")
            return 0  # MPS has no index concept
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
            print("\n>> Using CPU\n")
            return 0

    # --- No GPU at all ---
    if not gpus:
        print("\nNo GPU detected — falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return 0

    # --- GPU(s) found (CUDA desktop or Jetson) ---
    n = len(gpus)
    platform_tag = " (Jetson — unified memory)" if _is_jetson else ""
    print(f"\nFound {n} CUDA GPU(s){platform_tag}:\n")
    for idx, name, mem in gpus:
        if mem > 0:
            print(f"  [{idx}]  {name}  ({mem / 1024:.1f} GB)")
        else:
            print(f"  [{idx}]  {name}")
    print(f"\n  [c]  CPU (no GPU acceleration)")

    while True:
        choice = input(
            f"\nSelect GPU index [0-{n-1}] or 'c' for CPU (default: 0): "
        ).strip()
        if choice == "":
            sel = 0
            break
        elif choice.lower() == "c":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("\n>> Using CPU\n")
            return 0
        elif choice.isdigit() and 0 <= int(choice) < n:
            sel = int(choice)
            break
        else:
            print(f"  Invalid choice. Enter 0-{n-1} or 'c'.")

    physical_index = gpus[sel][0]

    # On Jetson there is only one GPU and CUDA_VISIBLE_DEVICES may not be
    # needed, but setting it is harmless and keeps behaviour consistent.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    print(f"\n>> Selected GPU {physical_index}: {gpus[sel][1]}")
    print(f"   (CUDA_VISIBLE_DEVICES={physical_index} — this GPU is now cuda:0)\n")
    return physical_index
