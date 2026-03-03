# Config
NVCC ?= nvcc
TARGET ?= mk_llama
SRC ?= llama.cu
CONFIG ?= Release

# Get Python headers and version
PYBIND_INC = $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Set compiler flags
NVCC_FLAGS = -DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math \
    -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose \
    -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda \
    -lcudadevrt -lcudart_static -lcublas -lineinfo -I${THUNDERKITTENS_ROOT}/include \
    -I${MEGAKERNELS_ROOT}/include $(PYBIND_INC) -shared -fPIC \
	-DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a

# Add debug flags
ifeq ($(CONFIG), Debug)
    NVCC_FLAGS += -DTIMING_RECORD_ENABLED
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCC_FLAGS) -o $(TARGET)$(EXT_SUFFIX)

# Clean target
clean:
	rm -f $(TARGET)$(EXT_SUFFIX)
