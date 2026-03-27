# Build any target in repo

# Build target
TARGET ?= mk_llama

# Key paths
export MEGAKERNELS_ROOT = $(CURDIR)
export THUNDERKITTENS_ROOT = $(MEGAKERNELS_ROOT)/ThunderKittens
export BUILD_DIR = $(MEGAKERNELS_ROOT)/build
export PYTHON = uv run python

# Default target
all: $(TARGET)

mk_llama:
	$(MAKE) -C demos/low-latency-llama mk_llama

mk_llama_3b:
	$(MAKE) -C demos/low-latency-llama mk_llama_3b

mk_mixtral:
	$(MAKE) -C demos/mixtral mk_mixtral

# Clean target
clean:
	$(MAKE) -C demos/low-latency-llama clean
	$(MAKE) -C demos/low-latency-llama-3b clean
	$(MAKE) -C demos/mixtral clean
