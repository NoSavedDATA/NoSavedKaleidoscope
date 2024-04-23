# Culang

Culang is a language created using LLVM and CUDA :)

## Install

**Pre-requisites**:

- WSL 2

- Install CUDA toolkit 12.0 [here](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local), then:

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

- Test using `culang`, type `1+1;`, it should return `2.00`.

---

## Compiler

CUDA:
```bash
clang++ -g -O3 -rdynamic toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -lcudart_static -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -o bin/culang
```

 > `dl.lib` e `rt.lib` (isso é uma tentativa de adicionar as bibliotecas `dl.lib` ou `rt.lib` ao linker com `-ldl` e `-lrt`)

---

## CU Files

```bash
clang++ test.cu --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -lcudart_static -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
```
