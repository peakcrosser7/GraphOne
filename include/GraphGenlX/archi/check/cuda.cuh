#pragma once

#include "GraphGenlX/debug/debug.hpp"

namespace graph_genlx {

template <typename T>
void CheckCudaErr(T result, char const *const func, const char *const file,
           int const line) {
#ifdef DEBUG_CUDA
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    abort();
  }
#endif
}

#define checkCudaErrors(val) CheckCudaErr((val), #val, __FILE__, __LINE__)

} // namespace graph_genlx