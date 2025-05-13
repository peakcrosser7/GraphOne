#pragma once


#define CUDA_CHECK(call) do {                                               \
    cudaError err = (call);                                                 \
    if (cudaSuccess != err) {                                               \
        fprintf(stderr, "CUDA error in file '%s' in line %d : %s (%d)\n",     \
                __FILE__, __LINE__, cudaGetErrorString(err), err);            \
        exit(EXIT_FAILURE);                                                   \
    } } while (0)


#define CUSPARSE_CHECK(func) do {                                               \
    cusparseStatus_t status = (func);                                           \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
        fprintf(stderr, "CUSPARSE API error in file '%s' in line %d : %s (%d)\n", \
                __FILE__, __LINE__, cusparseGetErrorString(status), status);     \
        exit(EXIT_FAILURE);                                        \
    } } while (0)
