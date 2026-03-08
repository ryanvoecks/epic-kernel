# Build any target in repo

# Build target
TARGET ?= mk_llama mk_llama_3b

# Key paths
export MEGAKERNELS_ROOT = $(CURDIR)
export THUNDERKITTENS_ROOT = $(MEGAKERNELS_ROOT)/ThunderKittens
export PYTHON = uv run python

# Default target
all: $(TARGET)

mk_llama:
	$(MAKE) -C demos/low-latency-llama mk_llama

mk_llama_3b:
	$(MAKE) -C demos/low-latency-llama mk_llama_3b

# Clean target
clean:
	$(MAKE) -C demos/low-latency-llama clean
	$(MAKE) -C demos/low-latency-llama-3b clean
