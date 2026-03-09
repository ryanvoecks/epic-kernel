#pragma once

#ifdef LLAMA_3B
#include "llama_3b.cuh"
using llama_globals = llama_3b_globals;
#else
#include "llama_1b.cuh"
using llama_globals = llama_1b_globals;
#endif
