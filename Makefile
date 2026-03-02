# Build any target in repo

# Config
TARGET ?= mk_llama

# Key paths
export MEGAKERNELS_ROOT = $(CURDIR)
export THUNDERKITTENS_ROOT = $(MEGAKERNELS_ROOT)/ThunderKittens
export PYTHON = uv run python

# Default target
all: $(TARGET)

mk_llama:
	$(MAKE) -C demos/low-latency-llama

# Clean target
clean:
	$(MAKE) -C demos/low-latency-llama clean
