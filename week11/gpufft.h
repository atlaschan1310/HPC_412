#include "blitz/array.h"
#include <cuda.h>
#include <cufft.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

void compute_fft_2D_R2C(blitz::Array<float, 2> &grid, void *gpu_slab);
void compute_fft_2D_R2C_Stream(blitz::Array<float,2>& grid, void* gpu_slab, cufftHandle* plan, void* workArea, cudaStream_t* stream);
void *allocate_cuda_slab(size_t nGrid);
void* allocate_cuda_size(size_t size);
void* allocate_cuda_slab_Stream(size_t nGrid, cudaStream_t* stream);
void destroy_cuda_data(void* data);
void destroy_cuda_data_Stream(void* data, cudaStream_t* stream);
