# Config
NVCC ?= nvcc
GPU ?= B200
TARGET = mk_llama
SRC = llama.cu

# Get the extension suffix directly from the active Python interpreter
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math \
    -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose \
    -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda \
    -lcudadevrt -lcudart_static -lcublas -lineinfo -I${THUNDERKITTENS_ROOT}/include \
    -I${MEGAKERNELS_ROOT}/include \
    $(shell python3 -m pybind11 --includes) -shared -fPIC

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(EXT_SUFFIX)

# Clean target
clean:
	rm -f $(TARGET)*.so
