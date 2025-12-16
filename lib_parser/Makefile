CXX := clang++-19
CXXFLAGS := -O3 -rdynamic -march=native -fno-exceptions
# CXXFLAGS := -O0 -g -rdynamic
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
CXXFLAGS += $(LLVM_CXXFLAGS) -mavx -w
LDFLAGS := $(LLVM_LDFLAGS) -static-libstdc++ -static-libgcc
LIBS := $(LLVM_LIBS) $(LLVM_SYSTEM_LIBS) $(SYSTEM_LIBS)



# Directories
LIB_PARSER_OBJ_DIR = ../lib_parser_obj
LIB_PARSER_SRC_DIR = ./
OBJ_DIR = obj
SRC_DIR = src
LIB_DIR := obj_static



# C++ Source and Object Files
CXX_SRC = $(shell find ./ -name "*.cpp")
CXX_OBJ = $(CXX_SRC:./%.cpp=./%.o)


# Lib Parser Object Files
LIB_PARSER_SRC = $(shell find $(LIB_PARSER_SRC_DIR) -name "*.cpp")


PARSED_LIBS = $(shell find ./parsed_libs -name "*.txt")





# Executable name
LIB_PARSER := ./lib_parser.o
OBJ := LibParser

.PHONY: prebuild

BUILD_FLAG := .build_flag






$(shell mkdir -p $(LIB_PARSER_OBJ_DIR);)




all: prebuild $(LIB_PARSER) check_done



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
	rm $(LIB_PARSER) $(PARSED_LIBS)

# Track dependencies
-include $(CXX_OBJ:.o=.d)
