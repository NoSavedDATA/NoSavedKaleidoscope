# No Saved Kaleidoscope Compiler

NSK is a LLVM/C++ programming language. 

All the code is open sourced.


<div align="center">
  <img src="assets/Logo1.jpg" alt="Logo" width="260" height="260">
</div>

Features: 
- 100% jitted interpreter (no function nor loop recompiling);
- Dense and Convolutional Neural Network training;
- Parallel coding with finish async expressions.

## Install

**Pre-requisites**:

- WSL 2

- clang version 19;

- Install CUDA toolkit 12.0 [here](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local);

- Install [cuDNN](https://developer.nvidia.com/cudnn-downloads) 9.1.1;

- Install OpenCV:
  
```bash
sudo apt install libopencv-dev
```

- Install LLVM dependencies:
  
```bash
chmod +x *.sh
sudo ./llvm.sh 19 all

sudo apt-get install llvm clang zlib1g-dev libzstd-dev
```
- Add commands `culang` and `cumpile` to `PATH`:

```bash
./alias.sh
```

- Update submodules:

```bash
./update_submodules.sh
```

- Test using `culang`, type `1+1;`, it should return `2.00`.

---

## Compiler

CUDA:
```bash
clang++ -g -O3 -rdynamic toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -I"/usr/local/cuda-12/include" -lcudart_static  -lcublas -lcublasLt -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -flto -finline-functions -funroll-loops -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -w -o bin/nsk
```

 > `dl.lib` e `rt.lib` (isso é uma tentativa de adicionar as bibliotecas `dl.lib` ou `rt.lib` ao linker com `-ldl` e `-lrt`)

---

## CU Files

```bash
clang++ test.cu --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -lcudart_static -lcublas -lcublasLt -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -w
```


## Todos:

- Map var names to hash function.

## Awards:

<div align="center">
  <img src="assets/Screenshot_2.jpg" alt="Logo" width="350" height="350">
</div>
