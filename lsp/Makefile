CXX := clang++-19
# CXXFLAGS := -O3 -rdynamic
CXXFLAGS := -O0 -g -rdynamic -fPIC
LLVM_CONFIG := llvm-config-19 --link-static --libs core orcjit native
SYSTEM_LIBS := -ldl -lrt -pthread
# OTHER_FLAGS := -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -flto -finline-functions -funroll-loops -fsanitize=address -w
OTHER_FLAGS := -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -flto -finline-functions -funroll-loops -w


# Get LLVM flags (must be from static LLVM build)
LLVM_CXXFLAGS := $(shell $(LLVM_CONFIG) --cxxflags)
LLVM_LDFLAGS := $(shell $(LLVM_CONFIG) --ldflags)
LLVM_SYSTEM_LIBS := $(shell $(LLVM_CONFIG) --system-libs)
LLVM_LIBS := $(shell $(LLVM_CONFIG) --libs core orcjit native)

# Combine all flags
CXXFLAGS += $(LLVM_CXXFLAGS) -DNODE_ADDON_API_DISABLE_CPP_EXCEPTIONS -mavx -w
LDFLAGS := $(LLVM_LDFLAGS) -static-libstdc++ -static-libgcc
LIBS := $(LLVM_LIBS) $(LLVM_SYSTEM_LIBS) $(SYSTEM_LIBS)


NODE_INCLUDES := $(shell node -p "require('node-addon-api').include" | xargs)
CXXFLAGS += -I/usr/include/node/node_api.h -I$(NODE_INCLUDES)

NODE_VERSION := $(shell node -p "process.versions.node")
NODE_HEADERS := $(HOME)/.cache/node-gyp/$(NODE_VERSION)/include/node

CXXFLAGS += -I$(NODE_HEADERS)




# Directories
LIB_PARSER_OBJ_DIR = lib_parser_obj
LIB_PARSER_SRC_DIR = ../lib_parser
OBJ_DIR = obj
BIN_DIR = bin
SRC_DIR = ../src
LIB_DIR := obj_static



# C++ Source and Object Files
CXX_SRC = $(shell find $(SRC_DIR) -name "*.cpp")
CXX_OBJ = $(CXX_SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CXX_DIR = $(sort $(dir $(CXX_OBJ)))

OBJ_DIRS := $(sort $(CXX_DIR))


# Lib Parser Object Files
LIB_PARSER_SRC = $(shell find $(LIB_PARSER_SRC_DIR) -name "*.cpp")






# Executable name
LIB_PARSER := ../bin/lib_parser.o
OBJ := lsp_parser.node
SRC := lsp_parser.cpp

.PHONY: prebuild

BUILD_FLAG := .build_flag



$(info var is: ${OBJ_DIRS})
$(foreach dir, $(OBJ_DIRS), \
  $(info var is: $(dir)) \
  $(shell mkdir -p $(dir)); \
)

$(shell mkdir -p $(BIN_DIR);)
$(shell mkdir -p $(LIB_DIR);)


$(shell mkdir -p $(LIB_PARSER_OBJ_DIR);)




all: prebuild $(CXX_OBJ) $(OBJ) check_done


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | prebuild
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<


$(OBJ): $(SRC) $(CXX_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SRC) $(CXX_OBJ) $(LIBS) $(OTHER_FLAGS) -shared -MMD -MP -o $(OBJ) 
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
-include $(CXX_OBJ:.o=.d)
