## Pygame / Triton Segmentation Fault — Investigation Summary

### Problem

SYNGrid was experiencing a native segmentation fault during evaluation when Pygame rendering was enabled. Evaluation without rendering (`render_mode=None`) worked correctly.

The crash initially appeared to be related to SB3/PPO model loading, but systematic isolation showed that the problem was caused by the interaction between Pygame's graphics stack and Triton's native library.

### Environment

* Pop!_OS
* Hybrid graphics: AMD integrated GPU + NVIDIA RTX 3060 Laptop GPU
* NVIDIA driver: 580.173.02
* CUDA: 13.0
* PyTorch: 2.13.0+cu130
* Triton: 3.7.1
* Pygame: 2.6.1
* SDL: 2.28.4
* Stable-Baselines3: 2.9.0
* Mesa: 25.1.5
* System LLVM: `libLLVM-15.so.1`

### Investigation

Several components were tested independently:

* PyTorch import and CPU tensors — worked
* CUDA tensors — worked
* Triton import — worked
* Triton + CUDA synchronization — worked
* SB3/PPO import — worked
* Pygame initialization — worked
* Pygame + PyTorch/CUDA — worked
* PPO checkpoint loading — worked
* The same checkpoint loaded successfully in a minimal program
* Evaluation without rendering — worked
* Evaluation with `human` rendering — crashed
* Evaluation with `rgb_array` rendering — crashed

The critical finding was that `pygame.display.set_mode()` triggered the problem when Triton was subsequently loaded.

The following worked:

```python
import triton
import pygame

pygame.init()
pygame.display.set_mode((640, 480))
```

while this crashed:

```python
import pygame

pygame.init()
pygame.display.set_mode((640, 480))

import triton
```

### GDB findings

GDB showed that the actual crash occurred inside:

```text
triton/_C/libtriton.so
```

during native library initialization:

```text
_GLOBAL__sub_I_PassBuilder.cpp
```

The top frame was an invalid `free()` inside LLVM's `DenseMap` implementation.

This established that the crash was not a normal Python exception or SYNGrid logic error. It was a native-library crash occurring while Triton's embedded LLVM/MLIR code was being initialized.

### Graphics Stack

`glxinfo` showed that the normal OpenGL renderer was:

```text
OpenGL vendor:   AMD
OpenGL renderer: AMD Radeon Graphics (radeonsi)
OpenGL version:  4.6 ... Mesa 25.1.5
```

Despite the RTX 3060 being active and used by Xorg, OpenGL rendering in the hybrid configuration was using the integrated AMD GPU.

`LD_DEBUG=libs` showed that after Pygame initialized the display, the AMD/Mesa graphics stack loaded:

```text
libLLVM-15.so.1
libgallium...
libGLX_mesa.so.0
```

before Triton's:

```text
libtriton.so
```

was loaded.

This ordering resulted in the crash.

### NVIDIA Offload Test

NVIDIA PRIME offloading was tested manually:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia glxinfo
```

This successfully reported:

```text
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: NVIDIA GeForce RTX 3060 Laptop GPU/PCIe/SSE2
OpenGL version string: 4.6.0 NVIDIA 580.173.02
```

The critical test was then:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python3 -c "import pygame; pygame.init(); pygame.display.set_mode((640,480)); import triton; print('OK')"
```

This completed successfully.

Finally, the actual SYNGrid evaluation was run with NVIDIA offload:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python3 -m syn_grid
```

The evaluation completed successfully without the segmentation fault.

### Conclusion

The most likely cause is a native-library compatibility/initialization conflict between Triton's embedded LLVM/MLIR runtime and the LLVM/Mesa graphics stack loaded when Pygame uses the integrated AMD GPU.

The important distinction is:

```text
PyTorch → Triton → embedded LLVM/MLIR
Pygame → SDL → X11 → AMD/Mesa → system LLVM 15
```

When AMD/Mesa is initialized first, Triton's native initialization can crash.

When Pygame is forced to use the NVIDIA OpenGL stack:

```text
Pygame → NVIDIA OpenGL → RTX 3060
```

the crash does not occur.

This does **not** mean that SYNGrid fundamentally requires an NVIDIA GPU. It indicates a specific compatibility problem in this particular hybrid AMD/Mesa + Triton environment.

### Current Workaround

A shell alias was created:

```bash
# Run SYNGrid evaluation using the NVIDIA GPU for OpenGL rendering
alias syn-grid='__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 -m syn_grid'
```

After reloading `.bashrc`:

```bash
source ~/.bashrc
```

SYNGrid can be started simply with:

```bash
syn-grid
```

The system can remain in Pop!_OS **Hybrid graphics mode**; there is no need to switch the entire system to NVIDIA mode.

### Future Consideration

For the current development/reproduction package, the NVIDIA-offload workaround is considered sufficient.

Before a future `1.0.0` release, it may be worth investigating whether the underlying Triton/Mesa/LLVM conflict can be eliminated or whether a more portable solution can be implemented.

For now, the segmentation fault investigation is considered resolved.
