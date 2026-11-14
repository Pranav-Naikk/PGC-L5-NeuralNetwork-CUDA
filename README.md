# Experiment 5: Neural Network Layers in CUDA

A CUDA C++ implementation of a two-layer feed-forward neural network using custom GPU kernels. The program performs matrix multiplication with bias, ReLU activation, and sigmoid activation, then compares CPU and GPU forward-pass results and execution time.

## Experiment objectives

- Implement matrix multiplication with bias using a custom CUDA kernel.
- Implement ReLU and sigmoid activation kernels.
- Execute the same forward pass on the CPU and GPU.
- Validate numerical agreement between CPU and GPU outputs.
- Measure CPU time, GPU kernel-only time, GPU end-to-end time, and speedup.

## Network architecture

The program performs inference (forward propagation) only. It does not train the network and does not implement loss calculation, an optimizer, or backpropagation.

```text
Input X (256 × 128)
       |
       v
Matrix multiplication + bias (X × W1 + b1)
       |
       v
ReLU
       |
       v
Hidden activation (256 × 64)
       |
       v
Matrix multiplication + bias (A1 × W2 + b2)
       |
       v
Sigmoid
       |
       v
Output (256 × 10)
```

### Configuration

| Parameter | Value |
|---|---:|
| Batch size | 256 |
| Input size | 128 |
| Hidden size | 64 |
| Output size | 10 |
| Timing repetitions | 20 |

Input data, weights, and biases are generated deterministically using sine and cosine functions so the CPU and GPU process identical values.

## CUDA kernels

| Kernel | Operation |
|---|---|
| `matrixMulBiasKernel` | Computes a matrix product and adds the output-column bias. Each CUDA thread computes one output element. |
| `reluKernel` | Applies ReLU to one array element per thread. |
| `sigmoidKernel` | Applies sigmoid to one array element per thread. |

The GPU forward pass launches the kernels in dependency order: matrix multiplication with bias → ReLU → matrix multiplication with bias → sigmoid.

## Requirements

- Windows 10/11 64-bit
- NVIDIA CUDA-capable GPU and NVIDIA driver
- CUDA Toolkit with `nvcc`
- Visual Studio 2022 Build Tools (or compatible MSVC C++ build tools)
- x64 Developer Command Prompt with `cl.exe` available

## Build and run (Windows)

Open **x64 Native Tools Command Prompt** or **Developer Command Prompt** so that the Microsoft compiler is available.

1. Create and enter a working directory:

   ```cmd
   mkdir "%USERPROFILE%\\Documents\\Experiment5_CUDA"
   cd /d "%USERPROFILE%\\Documents\\Experiment5_CUDA"
   ```

2. Place `neural_network.cu` in this directory.

3. Compile for the NVIDIA RTX 2050 (Compute Capability 8.6):

   ```cmd
   nvcc -O2 -arch=sm_86 neural_network.cu -o neural_network.exe
   ```

4. Run:

   ```cmd
   neural_network.exe
   ```

> **GPU architecture note:** The `-arch=sm_86` option targets the RTX 2050 used for the recorded run. For a different GPU, check its compute capability and adjust the architecture flag accordingly.

## Measured results

The following values are from one successful run on an **NVIDIA GeForce RTX 2050 (4 GB)**.

| Measurement | Observed result |
|---|---:|
| CPU average forward-pass time | 2.083975 ms |
| GPU kernel-only time | 0.064870 ms |
| GPU end-to-end time | 0.184795 ms |
| Kernel-only speedup | 32.125205× |
| End-to-end speedup | 11.277226× |
| Maximum absolute CPU/GPU output difference | 0.000000 |
| Kernel validation | All three tests passed |
| CPU/GPU output validation | Passed |

Speedup is calculated as:

- **Kernel-only speedup** = CPU average time ÷ GPU kernel-only time
- **End-to-end speedup** = CPU average time ÷ GPU end-to-end time

Kernel-only timing measures repeated GPU kernel execution with inputs and weights already resident on the device. End-to-end timing includes host-to-device transfers, kernel execution, synchronization, and device-to-host output transfer. Timing values depend on hardware, driver, compiler, and system load; results from other systems will differ.

## Validation

The program includes small known-answer tests for each CUDA kernel:

- Matrix multiplication with bias: expected output `[[23, 27], [50, 63]]`
- ReLU: input `[-2, 0, 3, -4, 5]`, expected output `[0, 0, 3, 0, 5]`
- Sigmoid: input `[-2, 0, 2]`, expected values approximately `[0.119203, 0.5, 0.880797]`

The full forward pass compares CPU and GPU output arrays and reports the maximum absolute difference. A result is considered matching when the difference is below `1e-4`.

## Repository contents

Expected project file:

```text
PGC-L5-NeuralNetwork-CUDA/
├── neural_network.cu
└── README.md
```

## Notes

- This is a learning/laboratory implementation using custom CUDA kernels, not a production neural-network framework.
- The measured performance values above are observations from the stated RTX 2050 run, not guaranteed benchmark results.
