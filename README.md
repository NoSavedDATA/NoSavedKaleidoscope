# No Saved Kaleidoscope Compiler

Official repository of: [https://arxiv.org/abs/2409.11600](https://arxiv.org/abs/2409.11600)

NSK is a LLVM/C++ programming language. 

All the code is open sourced.


<div align="center">
  <img src="assets/kaleidoscope.jpg" alt="Logo" width="260" height="260">
</div>

Features: 
- 100% jitted interpreter (no function nor loop recompiling);
- Trains ResNets and LSTMs;
- Automatic differentiation;
- Parallel coding with finish async expressions.

Limitations:
- It lacks modern network architectures support/testing, like GANs and Diffusion Models.

## Install

**Pre-requisites**:

- WSL 2 or Ubuntu 22

- clang version 19;

- Install CUDA toolkit 12.0 [here](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local);

- Install [cuDNN](https://developer.nvidia.com/cudnn-downloads) 9.4.0;

- Install OpenCV, cBLAS and LLVM dependencies:
  
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x *.sh
sudo ./llvm.sh 19 all

sudo apt-get install llvm clang zlib1g-dev libzstd-dev libeigen3-dev libopencv-dev
```
- Add commands `nsk` to `PATH`:

```bash
./alias.sh
```

- Update submodules:

```bash
./update_submodules.sh
```

---

## Compiler

- Run the C++ executable to generate the compiler executable.

CUDA:
```bash
clang++ -g -O3 -rdynamic toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path="/usr/local/cuda-12.1" --cuda-gpu-arch=sm_89 -L"/usr/local/cuda-12.1/lib64" -I"/usr/local/cuda-12.1/include" -I/usr/include/eigen3 -lcudart_static -lcublas -lcublasLt -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -flto -finline-functions -funroll-loops -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -w -o bin/nsk
```
- Install and use clang++-19 if you get an opm.h error;

- Test it by typing `nsk` at the terminal. Then, type `1+1;`, it should return `2.00`;

- Refer to the samples and experiments folders for example codes.

---

## CU Files

```bash
clang++ test.cu --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -lcudart_static -lcublas -lcublasLt -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -w
```


## Todos:

- Map var names to hash function;
- C++ vs Python vs NSK speed benchmarks;
- NSK vs PyTorch benchmarks.


## Changelogs

0.1
  - It is now possible to code custom datasets (there is no str dict yet);
  - Batch Norm and more activation functions support;
  - Embedding module added;
  - Fused LSTM kernels.

0.01
  - Launched Cifar and MNIST like hardcoded datasets for neural network training;
  - MLP and Conv2d support;
  - Neural networks have decent expressivity, but only single-path non-residual architectures are supported.

