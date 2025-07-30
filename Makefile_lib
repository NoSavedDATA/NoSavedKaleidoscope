#clang++-19 -g -O3 -rdynamic toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path="/usr/local/cuda-12.1" --cuda-gpu-arch=sm_89
#-L"/usr/local/cuda-12.1/lib64" -I"/usr/local/cuda-12.1/include" -I/usr/include/eigen3 -lcudart_static -lcublas -lcublasLt -ldl -lrt -pthread
#-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -flto -finline-functions -funroll-loops -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -w -o bin/nsk

CXX := clang++-19
NXX := nvcc
CXXFLAGS := -g -O3 -rdynamic -fPIC
LLVM_CONFIG := llvm-config-19
CUDA_PATH := /usr/local/cuda-12.1
CUDA_ARCH := sm_89
CUDA_ARCH_NVCC := -arch=sm_89
EIGEN_INCLUDE := /usr/include/eigen3
OPENCV_LIBS := -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
CUDA_LIBS := -lcudart_static -lcublas -lcublasLt -lcudnn
SYSTEM_LIBS := -ldl -lrt -pthread
OTHER_FLAGS := -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -flto -finline-functions -funroll-loops -w

# Get LLVM flags
LLVM_CXXFLAGS := $(shell $(LLVM_CONFIG) --cxxflags)
LLVM_LDFLAGS := $(shell $(LLVM_CONFIG) --ldflags)
LLVM_SYSTEM_LIBS := $(shell $(LLVM_CONFIG) --system-libs)
LLVM_LIBS := $(shell $(LLVM_CONFIG) --libs core orcjit native)


# CUDA flags
CUDA_CXXFLAGS := -I$(CUDA_PATH)/include --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=$(CUDA_ARCH)
CUDA_LDFLAGS := -L$(CUDA_PATH)/lib64

# Combine all flags
CXXFLAGS += $(LLVM_CXXFLAGS) $(CUDA_CXXFLAGS) -I$(EIGEN_INCLUDE) -mavx -w
LDFLAGS := $(LLVM_LDFLAGS) $(CUDA_LDFLAGS)
LIBS := $(LLVM_LIBS) $(LLVM_SYSTEM_LIBS) $(CUDA_LIBS) $(SYSTEM_LIBS) $(OPENCV_LIBS)

NXXFLAGS := $(CUDA_ARCH_NVCC) -I$(EIGEN_INCLUDE) -Xptxas=-v

NVCCFLAGS := -g -lineinfo \
             -Xcompiler -fPIC \
             -Xcompiler -rdynamic \
             -arch=$(CUDA_ARCH) \
             -I$(SRC_DIR) -I$(EIGEN_INCLUDE) \
             -I$(CUDA_PATH)/include \
             $(OTHER_FLAGS)

# Directories
OBJ_DIR = obj
SRC_DIR = src


# CUDA Source and Object Files
CU_SRC = $(shell find $(SRC_DIR) -name "*.cu")
CU_OBJ = $(CU_SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CU_DIR = $(sort $(dir $(CU_OBJ)))

CUH_SRC = $(shell find $(SRC_DIR) -name "*.cuh")
CUH_OBJ = $(CUH_SRC:$(SRC_DIR)/%.cuh=$(OBJ_DIR)/%.o)
CUH_DIR = $(sort $(dir $(CUH_OBJ)))


# C++ Source and Object Files
CXX_SRC = $(shell find $(SRC_DIR) -name "*.cpp")
CXX_OBJ = $(CXX_SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CXX_DIR = $(sort $(dir $(CXX_OBJ)))

OBJ_DIRS := $(sort $(CU_DIR) $(CXX_DIR) $(CUH_OBJ))







# Executable name
SO_FILE := lib.so


BUILD_FLAG := .build_flag



$(info var is: ${OBJ_DIRS})
$(foreach dir, $(OBJ_DIRS), \
  $(info var is: $(dir)) \
  $(shell mkdir -p $(dir)); \
)


$(info objects: $(CU_OBJ) sources: $(CU_SRC))





all: $(CU_OBJ) $(CXX_OBJ) $(SO_FILE) check_done


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -MMD -MP -c -o $@ $<
	
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp 
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<


$(SO_FILE): $(CU_OBJ) $(CXX_OBJ)
	$(CXX) -shared $(CXXFLAGS) $(LDFLAGS) $(CU_OBJ) $(CXX_OBJ) $(LIBS) $(OTHER_FLAGS) -MMD -MP -o $(SO_FILE) -lcudart
	@echo "\033[1;32m\nBuild completed [✓]\n\033[0m"
	@touch $(BUILD_FLAG)



check_done:
	@if [ ! -f $(BUILD_FLAG) ]; then \
		echo "\n\n\033[1;33mNo changes found [ ]\n\033[0m"; \
	fi
	@rm -f $(BUILD_FLAG)

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

# Track dependencies
-include $(CU_OBJ:.o=.d) $(CXX_OBJ:.o=.d)
