dl.lib e rt.lib (isso é uma tentativa de adicionar as bibliotecas dl.lib ou rt.lib ao linker com -ldl e -lrt)

clang++ .\test.cu -o axpy.exe --cuda-path=$env:CUDA_PATH --cuda-gpu-arch=sm_75 -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" -lcudart_static -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH


=============================================================================================================================================================================

Run .cu file

clang++ .\test.cu --cuda-path=$env:CUDA_PATH --cuda-gpu-arch=sm_75 -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" -lcudart_static -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH


Run .cu ubuntu

clang++ test.cu --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -lcudart_static -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Compiler:

cpu:

clang++ -g -O3 -rdynamic lexer.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o compilador
clang++ -g -rdynamic lexer.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy

clang++ -g -rdynamic toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy

CUDA:

clang++ -g -O3 -rdynamic toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -lcudart_static -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -o compilador











=============================================================================================================================================================================
cmake .. -G "Visual Studio 16 2019" -DCMAKE_INSTALL_PREFIX=C:\some_folder -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64
cmake .. -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64
cmake --build . --target install
