# ---- Performance monitor (CPU, RAM, GPU util/mem/power if available) ----
import os
import sys
import psutil
import time
import platform
import subprocess
import re
import json
import threading

# Add project root so we can import gpu_utils
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gpu_utils import is_jetson as _is_jetson_func

# Try NVIDIA NVML for NVIDIA GPUs
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# Check if we're on macOS for Apple Silicon GPU monitoring
_IS_MACOS = platform.system() == "Darwin"

# Check if we're on a Jetson / Tegra platform
_IS_JETSON = _is_jetson_func()


def _get_macos_gpu_power_ioreg():
    """
    Get system power consumption on macOS using ioreg.
    Returns total system power in Watts (approximation from battery discharge or adapter).
    """
    try:
        # Get battery/power info
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "AppleSmartBattery"],
            capture_output=True, text=True, timeout=3
        )
        output = result.stdout
        
        # Try to extract power from PowerOutDetails (Watts field)
        # This represents current power draw
        watts_match = re.search(r'"Watts"\s*=\s*(\d+)', output)
        if watts_match:
            watts = int(watts_match.group(1))
            # Convert from milliwatts if needed (some systems report mW)
            if watts > 1000:
                watts = watts / 1000.0
            return watts
        
        # Alternative: calculate from current and voltage
        # InstantAmperage is negative when discharging
        amperage_match = re.search(r'"InstantAmperage"\s*=\s*(-?\d+)', output)
        voltage_match = re.search(r'"Voltage"\s*=\s*(\d+)', output)
        
        if amperage_match and voltage_match:
            amperage = abs(int(amperage_match.group(1))) / 1000.0  # mA to A
            voltage = int(voltage_match.group(1)) / 1000.0  # mV to V
            power = amperage * voltage
            return power if power > 0 else None
            
    except Exception as e:
        pass
    
    return None


def _get_macos_gpu_utilization():
    """
    Attempt to get GPU utilization on macOS.
    Uses Activity Monitor's GPU history via hidden API if available.
    """
    try:
        # Try using top to get GPU process info (limited but works)
        result = subprocess.run(
            ["ps", "-A", "-o", "%cpu,command"],
            capture_output=True, text=True, timeout=3
        )
        
        # Look for GPU-related processes (Metal, WindowServer, etc.)
        gpu_cpu = 0.0
        for line in result.stdout.split('\n'):
            if any(proc in line.lower() for proc in ['windowserver', 'metal', 'gpu']):
                try:
                    cpu_pct = float(line.strip().split()[0])
                    gpu_cpu += cpu_pct
                except:
                    pass
        
        # This is a rough approximation - GPU work shows up as WindowServer CPU
        # Clamp to 0-100 range
        return min(100.0, gpu_cpu) if gpu_cpu > 0 else None
        
    except Exception:
        pass
    
    return None


def _get_macos_thermal_pressure():
    """
    Get thermal pressure on macOS which indicates system load including GPU.
    """
    try:
        result = subprocess.run(
            ["sysctl", "machdep.xcpm.cpu_thermal_level"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            match = re.search(r':\s*(\d+)', result.stdout)
            if match:
                # Thermal level 0-100 indicates system thermal pressure
                return int(match.group(1))
    except Exception:
        pass
    return None


def _check_metal_gpu_available():
    """Check if Metal GPU is available on macOS."""
    if not _IS_MACOS:
        return False
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5
        )
        return "Metal" in result.stdout
    except Exception:
        return False


def _get_macos_gpu_info():
    """Get GPU name and core count on macOS."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        if displays:
            gpu = displays[0]
            name = gpu.get("sppci_model", "Unknown GPU")
            cores = gpu.get("sppci_cores", "?")
            return name, cores
    except Exception:
        pass
    return "Apple GPU", "?"


# ── Jetson / Tegra monitoring helpers ───────────────────────────────

def _parse_tegrastats_line(line: str) -> dict:
    """
    Parse a single line from ``tegrastats`` output into a dict.

    Example line (Jetson Orin NX):
      02-25-2026 09:54:28 RAM 5892/15656MB ... GR3D_FREQ 48% ...
      gpu@55.062C ... VDD_IN 6914mW/6914mW VDD_CPU_GPU_CV 1229mW/1229mW ...
    """
    info = {}

    # GPU utilisation: "GR3D_FREQ 48%"
    m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if m:
        info['gpu_util'] = float(m.group(1))

    # RAM used/total: "RAM 5892/15656MB"
    m = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
    if m:
        info['ram_used_mb'] = float(m.group(1))
        info['ram_total_mb'] = float(m.group(2))

    # GPU temperature: "gpu@55.062C"
    m = re.search(r'gpu@([\d.]+)C', line)
    if m:
        info['gpu_temp'] = float(m.group(1))

    # Total board power: "VDD_IN 6914mW/6914mW" (instant/average)
    m = re.search(r'VDD_IN\s+(\d+)mW', line)
    if m:
        info['power_mw'] = float(m.group(1))

    # CPU+GPU+CV power rail (more specific to compute)
    m = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
    if m:
        info['cpu_gpu_power_mw'] = float(m.group(1))

    return info


def _tegrastats_snapshot() -> dict:
    """
    Collect Jetson GPU metrics from sysfs (fast, < 1 ms) with tegrastats
    as a background fallback for power/temperature on first call.

    Returns a dict with gpu_util, gpu_temp, power_mw, etc.
    """
    info = {}

    # ── Fast path: read GPU load from sysfs (< 1 ms) ──
    try:
        for path in [
            "/sys/devices/platform/bus@0/17000000.gpu/load",
            "/sys/devices/platform/gpu.0/load",
            "/sys/devices/gpu.0/load",
        ]:
            if os.path.isfile(path):
                with open(path) as f:
                    val = int(f.read().strip())
                info['gpu_util'] = val / 10.0  # 0-1000 → 0-100
                break
    except Exception:
        pass

    # ── Fast path: read GPU temperature from thermal zone ──
    try:
        # Scan thermal zones for GPU
        for tz_dir in sorted(os.listdir("/sys/class/thermal/")):
            if not tz_dir.startswith("thermal_zone"):
                continue
            tz_path = f"/sys/class/thermal/{tz_dir}"
            try:
                with open(f"{tz_path}/type") as f:
                    tz_type = f.read().strip().lower()
                if "gpu" in tz_type:
                    with open(f"{tz_path}/temp") as f:
                        info['gpu_temp'] = int(f.read().strip()) / 1000.0
                    break
            except Exception:
                continue
    except Exception:
        pass

    # ── Fast path: read power from INA sensors (Jetson Orin) ──
    try:
        # Scan hwmon directories for INA3221-style sensors
        hwmon_base = "/sys/class/hwmon"
        if os.path.isdir(hwmon_base):
            for hd in os.listdir(hwmon_base):
                hpath = f"{hwmon_base}/{hd}"
                for i in range(1, 10):
                    label_f = f"{hpath}/in{i}_label"
                    curr_f = f"{hpath}/curr{i}_input"
                    volt_f = f"{hpath}/in{i}_input"
                    try:
                        if not os.path.isfile(label_f):
                            continue
                        with open(label_f) as f:
                            label = f.read().strip()
                        if label.startswith("VDD_IN"):
                            # INA3221: curr in mA, voltage in mV
                            if os.path.isfile(curr_f) and os.path.isfile(volt_f):
                                with open(curr_f) as f:
                                    curr_ma = float(f.read().strip())
                                with open(volt_f) as f:
                                    volt_mv = float(f.read().strip())
                                info['power_mw'] = (curr_ma * volt_mv) / 1000.0  # mA * mV / 1000 → mW
                                break
                    except Exception:
                        continue
                if 'power_mw' in info:
                    break
    except Exception:
        pass

    # ── Slow fallback: tegrastats for anything still missing ──
    if 'gpu_temp' not in info or 'power_mw' not in info:
        try:
            result = subprocess.run(
                ["tegrastats", "--interval", "200"],
                capture_output=True, text=True, timeout=1,
            )
            output = result.stdout
        except subprocess.TimeoutExpired as e:
            output = e.stdout if e.stdout else ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
        except Exception:
            output = ""

        if output:
            for line in output.strip().splitlines():
                parsed = _parse_tegrastats_line(line)
                if parsed:
                    # Only fill in missing fields
                    for k, v in parsed.items():
                        if k not in info:
                            info[k] = v
                    break

    return info


def _get_jetson_gpu_name() -> str:
    """Get the Jetson GPU name from nvidia-smi or device-tree."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        name = out.strip()
        if name:
            return name
    except Exception:
        pass
    # Fallback to device-tree model
    try:
        with open("/proc/device-tree/model", "r") as f:
            return f.read().strip().rstrip('\x00')
    except Exception:
        pass
    return "Jetson GPU"


class SystemMonitor:
    """
    Cross-platform system monitor for CPU, RAM, and GPU metrics.

    Supports two modes of operation:

    1. **Manual** – call :meth:`sample` explicitly (legacy behaviour).
    2. **Background** – call :meth:`start` before the workload and
       :meth:`stop` after it.  A daemon thread continuously samples at
       *interval_ms* (default 100 ms) so that short GPU bursts are not
       missed.  This is the recommended mode.

    Supports:
    - CPU/RAM: All platforms via psutil
    - GPU (NVIDIA desktop/server): Via pynvml
    - GPU (NVIDIA Jetson / Tegra): Via tegrastats + sysfs
    - GPU (macOS/Apple Silicon): Via ioreg and system tools
    """
    
    def __init__(self, gpu_index: int = 0, interval_ms: int = 100):
        self.proc = psutil.Process(os.getpid())
        self.gpu_index = gpu_index
        self.has_nvml = False
        self.has_jetson = False
        self.has_metal = False
        self.handle = None
        self.samples = []
        self.platform = platform.system()
        # Background sampling
        self._interval = interval_ms / 1000.0   # seconds
        self._bg_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.gpu_name = "Unknown"
        self.gpu_cores = "?"

        # Prime CPU percent (first call always returns 0.0)
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)

        # Check Jetson first — pynvml is present on Jetson but most queries
        # return "Not Supported", so we must use tegrastats/sysfs instead.
        if _IS_JETSON:
            self.has_jetson = True
            self.gpu_name = _get_jetson_gpu_name()
            print(f"[SystemMonitor] Jetson GPU detected: {self.gpu_name}")
            print(f"[SystemMonitor] Using tegrastats/sysfs for GPU metrics")

        # Try NVIDIA NVML for desktop/server GPUs (skip full init on Jetson)
        elif _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                # Verify that utilisation queries actually work (they don't on Jetson)
                pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.has_nvml = True
                name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                self.gpu_name = name
                print(f"[SystemMonitor] NVIDIA GPU detected: {self.gpu_name}")
            except Exception as e:
                self.has_nvml = False
        
        # Check for macOS Metal GPU
        if not self.has_nvml and not self.has_jetson and _IS_MACOS:
            self.has_metal = _check_metal_gpu_available()
            if self.has_metal:
                self.gpu_name, self.gpu_cores = _get_macos_gpu_info()
                print(f"[SystemMonitor] macOS GPU detected: {self.gpu_name} ({self.gpu_cores} cores)")
            else:
                print(f"[SystemMonitor] No GPU monitoring available on this system")
        elif not self.has_nvml and not self.has_jetson:
            print(f"[SystemMonitor] No GPU detected, GPU metrics disabled")

    def sample(self):
        """Take a sample of current system metrics."""
        # CPU & RAM
        # psutil.Process.cpu_percent() returns the sum across all cores on Linux
        # (e.g. 400% on a 4-core machine). Normalise to 0–100% range.
        n_cores = psutil.cpu_count(logical=True) or 1
        cpu_proc_raw = self.proc.cpu_percent(None)
        cpu_proc = cpu_proc_raw / n_cores
        cpu_sys = psutil.cpu_percent(None)  # already 0-100%
        ram_proc = self.proc.memory_info().rss / (1024**2)  # MB
        ram_sys = psutil.virtual_memory().percent

        gpu_util = gpu_mem = gpu_power = gpu_temp = None
        
        # NVIDIA GPU via NVML (desktop/server)
        if self.has_nvml:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                gpu_util = float(util.gpu)
                gpu_mem = mem.used / (1024**2)  # MB
                try:
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W
                except Exception:
                    pass
                try:
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(
                        self.handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass
            except Exception:
                pass
        
        # NVIDIA Jetson / Tegra via tegrastats + sysfs
        elif self.has_jetson:
            snap = _tegrastats_snapshot()
            gpu_util = snap.get('gpu_util')
            gpu_temp = snap.get('gpu_temp')
            # Power: prefer total board power, fall back to CPU+GPU rail
            power_mw = snap.get('power_mw') or snap.get('cpu_gpu_power_mw')
            if power_mw is not None:
                gpu_power = power_mw / 1000.0  # mW → W
            # Jetson uses unified memory; report process RSS as GPU-related mem
            gpu_mem = ram_proc

        # macOS Metal GPU
        elif self.has_metal:
            # Get GPU utilization estimate (from WindowServer/GPU processes)
            gpu_util = _get_macos_gpu_utilization()
            
            # Get power consumption from ioreg
            gpu_power = _get_macos_gpu_power_ioreg()
            
            # Memory: use process memory as approximation for GPU-related work
            # Apple Silicon shares memory between CPU and GPU
            gpu_mem = ram_proc  # Shared memory architecture
            
            # Temperature: try to get thermal pressure as proxy
            thermal = _get_macos_thermal_pressure()
            if thermal is not None:
                # Convert thermal level to approximate temperature
                # This is a rough estimate
                gpu_temp = 30 + (thermal * 0.5)  # Base 30°C + thermal pressure

        with self._lock:
            self.samples.append({
                "cpu_proc": cpu_proc,
                "cpu_sys": cpu_sys,
                "ram_proc_mb": ram_proc,
                "ram_sys_pct": ram_sys,
                "gpu_util": gpu_util,
                "gpu_mem_mb": gpu_mem,
                "gpu_power_w": gpu_power,
                "gpu_temp_c": gpu_temp,
        })

    def get_averages(self) -> dict:
        """Get average metrics from all samples."""
        def safe_avg(xs):
            xs = [x for x in xs if x is not None]
            return (sum(xs) / len(xs)) if xs else None

        with self._lock:
            snapshot = list(self.samples)

        return {
            "cpu_proc": safe_avg([s["cpu_proc"] for s in snapshot]),
            "cpu_sys": safe_avg([s["cpu_sys"] for s in snapshot]),
            "ram_proc_mb": safe_avg([s["ram_proc_mb"] for s in snapshot]),
            "ram_sys_pct": safe_avg([s["ram_sys_pct"] for s in snapshot]),
            "gpu_util": safe_avg([s["gpu_util"] for s in snapshot]),
            "gpu_mem_mb": safe_avg([s["gpu_mem_mb"] for s in snapshot]),
            "gpu_power_w": safe_avg([s["gpu_power_w"] for s in snapshot]),
            "gpu_temp_c": safe_avg([s["gpu_temp_c"] for s in snapshot]),
        }

    def report(self):
        """Print a summary report of collected metrics."""
        avgs = self.get_averages()
        
        print("\n----- Runtime System Metrics (averages) -----")
        print(f"GPU: {self.gpu_name}")
        if avgs["cpu_proc"] is not None:
            print(f"Process CPU usage:   {avgs['cpu_proc']:.1f}%")
        if avgs["cpu_sys"] is not None:
            print(f"System CPU usage:    {avgs['cpu_sys']:.1f}%")
        if avgs["ram_proc_mb"] is not None:
            print(f"Process RAM:         {avgs['ram_proc_mb']:.1f} MB")
        if avgs["ram_sys_pct"] is not None:
            print(f"System RAM used:     {avgs['ram_sys_pct']:.1f}%")

        if self.has_nvml or self.has_metal or self.has_jetson:
            if avgs["gpu_util"] is not None:
                print(f"GPU util:            {avgs['gpu_util']:.1f}%")
            else:
                print(f"GPU util:            N/A")
            if avgs["gpu_mem_mb"] is not None:
                print(f"GPU memory (used):   {avgs['gpu_mem_mb']:.1f} MB")
            if avgs["gpu_power_w"] is not None:
                print(f"GPU/System power:    {avgs['gpu_power_w']:.1f} W")
            else:
                print(f"GPU/System power:    N/A (connect to battery or check ioreg)")
            if avgs["gpu_temp_c"] is not None:
                print(f"GPU temperature:     {avgs['gpu_temp_c']:.1f} °C")
        else:
            print("GPU metrics:         (No GPU monitoring available)")
        
        print(f"Total samples:       {len(self.samples)}")
        print("---------------------------------------------\n")
    
    def gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self.has_nvml or self.has_metal or self.has_jetson

    # ── Background continuous sampling ──────────────────────────────

    def start(self):
        """Start background sampling thread.

        Samples are collected every *interval_ms* (set in __init__) and
        appended to ``self.samples`` under a lock.  Call :meth:`stop` when
        the workload is done.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def stop(self):
        """Stop background sampling and wait for the thread to finish."""
        self._stop_event.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
            self._bg_thread = None

    def _bg_loop(self):
        """Background thread entry-point: sample in a tight loop."""
        while not self._stop_event.is_set():
            self.sample()
            self._stop_event.wait(self._interval)

    def get_gpu_info_string(self) -> str:
        """Get a string describing the GPU monitoring status."""
        if self.has_nvml:
            return f"NVIDIA GPU: {self.gpu_name}"
        elif self.has_jetson:
            return f"Jetson GPU: {self.gpu_name} (unified memory)"
        elif self.has_metal:
            return f"Apple GPU: {self.gpu_name} ({self.gpu_cores} cores)"
        else:
            return "No GPU monitoring"
