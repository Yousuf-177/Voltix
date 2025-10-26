# GPU Initialization & Setup — README (NVIDIA, AMD (ROCm), Intel GPUs)

This guide explains how to **prepare your machine to run PyTorch-based training on a GPU** for the YOLO project (Team **Voltix**).
It covers step-by-step instructions for **NVIDIA (CUDA)**, **AMD Radeon (ROCm)**, and **Intel GPUs (XPU / oneAPI + Intel Extension for PyTorch)** and includes quick tests you can run to confirm everything is working.

> **Important:** GPU driver stacks and PyTorch/ROCm/oneAPI releases change over time. For the *most up-to-date* installation command for PyTorch and platform-specific instructions, always consult the official PyTorch or vendor pages referenced below. ([PyTorch][1])

---

## Quick checklist (before you start)

1. Determine your GPU vendor and model (`lspci` on Linux or Device Manager on Windows).
2. Make sure you have a compatible OS (many GPU stacks are best supported on Ubuntu LTS / Windows 10/11).
3. Back up important work — driver installs sometimes require reboots.
4. Have a Python virtual environment ready (recommended).

---

# A — NVIDIA GPUs (CUDA + PyTorch)

### Summary

If you have an NVIDIA GPU, install the **NVIDIA driver** + optional **CUDA toolkit** (PyTorch binary may include CUDA runtime so local nvcc is optional) and then install a PyTorch build compatible with your CUDA version. The PyTorch website has a selector that generates the exact pip/conda command for your platform and CUDA version. ([PyTorch][1])

### Steps (Linux / Windows)

1. **Check GPU & driver (pre-install):**

   * Linux:

     ```bash
     lspci | grep -i nvidia
     # After driver install:
     nvidia-smi
     ```
   * Windows: open *Device Manager* → Display adapters; run `nvidia-smi` in a terminal (if drivers installed).

2. **Install NVIDIA driver:**

   * Use the driver from NVIDIA (GeForce / Studio / Data Center) for your GPU model.
   * On Ubuntu, you can use:

     ```bash
     sudo apt update
     sudo ubuntu-drivers autoinstall
     sudo reboot
     ```
   * Or download drivers from NVIDIA website and follow the installer instructions.

3. **(Optional) Install CUDA toolkit** — only if you need local CUDA tooling (nvcc). Note: many PyTorch pip/conda packages include CUDA runtime; you *don’t* strictly need full CUDA toolkit to run GPU PyTorch code. See PyTorch selector for matching binaries. ([PyTorch][1])

4. **Install PyTorch with CUDA support**

   * Go to PyTorch “Get Started” page, pick your OS / Package / Language and the desired CUDA version, then copy the generated command and run it. Example (replace cu118 with the appropriate CUDA version from the selector):

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   * **Official reference & installer**: PyTorch local install selector. ([PyTorch][1])

5. **Verify in Python**

   ```python
   import torch
   print("torch.__version__:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   print("CUDA device count:", torch.cuda.device_count())
   if torch.cuda.is_available():
       print("Device name:", torch.cuda.get_device_name(0))
   ```

   * If `torch.cuda.is_available()` returns `True` and `nvidia-smi` shows the card, you're good.

6. **Troubleshooting**

   * If `torch.cuda.is_available()` is False but `nvidia-smi` works:

     * Reinstall PyTorch with the CUDA build that matches your driver/CUDA runtime.
     * Ensure PATH/LD_LIBRARY_PATH (Linux) includes CUDA libraries (if you installed CUDA toolkit).
     * Reboot after driver install.
   * If driver install fails, check kernel compatibility (Linux), Secure Boot settings, or use NVIDIA’s runfile/apt instructions.

---

# B — AMD Radeon GPUs (ROCm + PyTorch)

### Summary

AMD GPU acceleration in PyTorch is provided via the **ROCm** platform. You must install **ROCm** and then install a PyTorch build compiled for ROCm. ROCm support is primarily for certain Linux kernels and supported GPU families — check ROCm compatibility for your GPU. ([rocm.docs.amd.com][2])

### Steps (Linux recommended — ROCm is Linux focused)

1. **Check GPU compatibility**

   * Confirm your GPU model is supported by ROCm (see AMD ROCm docs).

2. **Install ROCm**

   * Follow AMD ROCm installation instructions for your distribution (Ubuntu recommended). The ROCm docs provide platform-specific commands. Example resources are in ROCm docs. ([rocm.docs.amd.com][2])

3. **Install PyTorch (ROCm build)**

   * Use the PyTorch “Get Started” selector and choose *ROCm* as the CUDA/ROCm option; copy the pip/conda command it gives. Example (pseudo):

     ```bash
     pip3 install torch --index-url https://download.pytorch.org/whl/rocmX.Y
     ```
   * Follow AMD docs (they also show recommended pip/conda commands for PyTorch on ROCm). ([rocm.docs.amd.com][3])

4. **Verify in Python**

   ```python
   import torch
   print("torch.__version__:", torch.__version__)
   print("ROCm / CUDA available via torch:", torch.cuda.is_available())
   print("Device count:", torch.cuda.device_count())
   if torch.cuda.is_available():
       print("Using device:", torch.device("cuda"))
   ```

   * ROCm-enabled PyTorch typically exposes devices through `torch.cuda` APIs; `torch.cuda.is_available()` should return `True` when ROCm is correctly installed and a supported GPU is present. ([rocm.docs.amd.com][2])

5. **Troubleshooting**

   * If `torch.cuda.is_available()` is False after ROCm install:

     * Verify `rocminfo` and `rocm-smi` commands are present and show the GPU.
     * Confirm ROCm version compatibility with your GPU and Linux kernel.
     * Check PyTorch ROCm wheel version matches the installed ROCm runtime.
   * There are known hardware/kernel compatibility issues for older GPUs or newer kernels — refer to ROCm docs and the community for workarounds. ([rocm.docs.amd.com][3])

---

# C — Intel GPUs (Xe / Arc) — oneAPI + Intel Extension for PyTorch (XPU)

### Summary

Intel provides a path to accelerate PyTorch on Intel GPUs via **oneAPI** toolkits and the **Intel Extension for PyTorch**. The PyTorch XPU backend (`torch.xpu`) is used to detect and run on supported Intel GPUs. You must install the Intel GPU drivers and oneAPI components, plus the Intel extension that integrates with PyTorch. ([PyTorch Documentation][4])

### Steps (Linux & WSL2 / Windows guidance)

1. **Check GPU & driver**

   * Linux:

     ```bash
     lspci | grep -i intel
     ```
   * Windows: check Device Manager for Intel® Iris/Xe/Arc entries.

2. **Install Intel GPU driver + oneAPI (required components)**

   * Follow Intel’s instructions to install the **Intel GPU driver** and the **Intel oneAPI Base Toolkit** (DPC++ compiler, oneMKL, etc.). The Intel Extension docs list required components. ([intel.github.io][5])

3. **Install Intel Extension for PyTorch**

   * Install the extension (version must match your PyTorch install). Example:

     ```bash
     pip install intel_extension_for_pytorch
     ```
   * For GPU acceleration on Intel GPUs, follow the Intel extension installation instructions (wheel files or oneAPI integrated installs). ([GitHub][6])

4. **Install a compatible PyTorch build**

   * Use the PyTorch instructions that mention XPU / Intel builds or use the Intel Extension’s guidance on compatible PyTorch versions. See Intel’s install guide for recommended commands. ([intel.github.io][5])

5. **Verify in Python (XPU check)**

   ```python
   import torch
   # XPU (Intel GPU) API
   print("torch.__version__:", torch.__version__)
   try:
       print("XPU available:", torch.xpu.is_available())
       if torch.xpu.is_available():
           print("XPU device count:", torch.xpu.device_count())
   except Exception as e:
       print("torch.xpu API not present or failed:", e)
   ```

   * `torch.xpu.is_available()` should return `True` when Intel GPU drivers + oneAPI + Intel Extension are correctly installed. ([PyTorch Documentation][7])

6. **Troubleshooting**

   * If `torch.xpu` is missing or returns `False`:

     * Make sure you installed the Intel GPU driver and activated any `oneAPI` environment (`source {ONEAPI_ROOT}/setvars.sh` on Linux). ([intel.github.io][5])
     * Check that the Intel Extension wheel matches your PyTorch version.
     * On Windows/WSL, ensure the correct driver for WSL is installed and WSL has GPU support enabled.

---

# D — Common verification script (works as final smoke test)

Save as `gpu_test.py` and run inside the environment you intend to train with:

```python
# gpu_test.py
import torch, sys
print("torch version:", torch.__version__)

# Check CUDA (NVIDIA / ROCm)
cuda_avail = torch.cuda.is_available()
print("torch.cuda.is_available():", cuda_avail)
if cuda_avail:
    try:
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA current device:", torch.cuda.current_device())
        # get device name safely
        print("CUDA device 0 name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("CUDA info error:", e)

# Check Intel XPU (if available in build)
if hasattr(torch, "xpu"):
    try:
        print("torch.xpu.is_available():", torch.xpu.is_available())
        if torch.xpu.is_available():
            print("torch.xpu.device_count():", torch.xpu.device_count())
    except Exception as e:
        print("torch.xpu check error:", e)
else:
    print("torch.xpu: Not available in this PyTorch build")

# Minimal GPU operation (if CUDA available)
if cuda_avail:
    device = torch.device("cuda")
    x = torch.randn(1024, 1024, device=device)
    print("Allocated tensor on GPU:", x.device)
    print("GPU test passed")
else:
    print("No CUDA GPU available — falling back to CPU.")
    sys.exit(1)
```

Run:

```bash
python gpu_test.py
```

---

# E — Troubleshooting checklist (short)

* `nvidia-smi` missing → NVIDIA driver not installed or PATH issue.
* `torch.cuda.is_available()` is False while `nvidia-smi` works → mismatched PyTorch/CUDA binary vs driver; reinstall PyTorch using the PyTorch selector. ([PyTorch][1])
* `rocminfo` / `rocm-smi` not found → ROCm not installed or incompatible kernel. ([rocm.docs.amd.com][2])
* `torch.xpu.is_available()` False → verify Intel GPU driver + oneAPI are installed and oneAPI environment activated. ([intel.github.io][5])

---

# F — Useful links & references (official)

* PyTorch — Get Started / Local Install selector (pick OS + CUDA/ROCm): [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) . ([PyTorch][1])
* AMD ROCm — PyTorch installation/help: [https://rocm.docs.amd.com](https://rocm.docs.amd.com) (PyTorch / ROCm instructions). ([rocm.docs.amd.com][2])
* Intel — Getting started on Intel GPU & Intel Extension for PyTorch: Intel docs and the Intel Extension repo. ([PyTorch Documentation][4])

---

## Notes & Recommendations

* For **hackathon** / quick demos: NVIDIA GPUs + CUDA are typically the least friction (drivers and PyTorch packages are mature).
* If you use **ROCm** (AMD) or **Intel XPU**, expect some additional compatibility checks and possibly kernel/driver matching. Always use the vendor docs to choose compatible package versions. ([rocm.docs.amd.com][2])
* Use a virtual environment (venv/conda) for each GPU stack to avoid version conflicts.

---
