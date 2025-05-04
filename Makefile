#clang++-19 -g -O3 -rdynamic toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path="/usr/local/cuda-12.1" --cuda-gpu-arch=sm_89
#-L"/usr/local/cuda-12.1/lib64" -I"/usr/local/cuda-12.1/include" -I/usr/include/eigen3 -lcudart_static -lcublas -lcublasLt -ldl -lrt -pthread
#-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -flto -finline-functions -funroll-loops -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -w -o bin/nsk

CXX := clang++-19
NXX := nvcc
CXXFLAGS := -g -O3 -rdynamic
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
CXXFLAGS += $(LLVM_CXXFLAGS) $(CUDA_CXXFLAGS) -I$(EIGEN_INCLUDE)
LDFLAGS := $(LLVM_LDFLAGS) $(CUDA_LDFLAGS)
LIBS := $(LLVM_LIBS) $(LLVM_SYSTEM_LIBS) $(CUDA_LIBS) $(SYSTEM_LIBS) $(OPENCV_LIBS)

NXXFLAGS := $(CUDA_ARCH_NVCC) -I$(EIGEN_INCLUDE) -Xptxas=-v

# Directories
LIB_PARSER_OBJ_DIR = lib_parser_obj
LIB_PARSER_SRC_DIR = lib_parser
OBJ_DIR = obj
BIN_DIR = bin
SRC_DIR = src
LIB_DIR := obj_static


# CUDA Source and Object Files
CU_SRC = $(shell find $(SRC_DIR) -name "*.cu")
CU_OBJ = $(CU_SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CU_DIR = $(sort $(dir $(CU_OBJ)))

# C++ Source and Object Files
CXX_SRC = $(shell find $(SRC_DIR) -name "*.cpp")
CXX_OBJ = $(CXX_SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CXX_DIR = $(sort $(dir $(CXX_OBJ)))

OBJ_DIRS := $(sort $(CU_DIR) $(CXX_DIR))


# Lib Parser Object Files
LIB_PARSER_SRC = $(shell find $(LIB_PARSER_SRC_DIR) -name "*.cpp")



# static libs (.a) for .o files
LIB_SUBDIRS := $(shell find $(OBJ_DIR) -mindepth 1 -maxdepth 1 -type d)
STATIC_LIBS := $(patsubst $(OBJ_DIR)/%, $(LIB_DIR)/%.a, $(LIB_SUBDIRS))



# Executable name
LIB_PARSER := bin/lib_parser.o
OBJ := bin/nsk
SRC := toy.cu

.PHONY: prebuild

BUILD_FLAG := .build_flag



$(info var is: ${OBJ_DIRS})
$(foreach dir, $(OBJ_DIRS), \
  $(info var is: $(dir)) \
  $(shell mkdir -p $(dir)); \
)

$(shell mkdir -p $(BIN_DIR);)
$(shell mkdir -p $(LIB_DIR);)

$(info objects: $(CU_OBJ) sources: $(CU_SRC))

$(shell mkdir -p $(LIB_PARSER_OBJ_DIR);)




all: prebuild $(OBJ) check_done

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | prebuild
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | prebuild
	$(CXX) $(CXXFLAGS) -c -o $@ $<


$(OBJ): $(SRC) $(CU_OBJ) $(CXX_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SRC) $(CU_OBJ) $(CXX_OBJ) $(LIBS) $(OTHER_FLAGS) -o $(OBJ) -lcudart
	@echo "\033[1;32m\nBuild completed [✓]\n\033[0m"
	@touch $(BUILD_FLAG)


prebuild: $(LIB_PARSER)
	@echo ">>> PREBUILD STEP <<<"
	$(shell bin/lib_parser.o;)

$(LIB_PARSER_OBJ_DIR)/%.o: $(LIB_PARSER_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(LIB_PARSER) : $(LIB_PARSER_SRC)
	$(CXX) $(CXXFLAGS) $(LIB_PARSER_SRC) -o $(LIB_PARSER)
	@echo "------------PREBUILD DONE DEON DEONDEON DEON ODENDEON"



check_done:
	@if [ ! -f $(BUILD_FLAG) ]; then \
		echo "\n\n\033[1;33mNo changes found [ ]\n\033[0m"; \
	fi
	@rm -f $(BUILD_FLAG)

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) $(LIB_PARSER_OBJ_DIR)

# Track dependencies
-include $(CU_OBJ:.o=.d) $(CXX_OBJ:.o=.d)
