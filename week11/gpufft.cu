#include "gpufft.h"

static const char *cufftGetErrorString(cufftResult cufft_error_type)
{

    switch (cufft_error_type)
    {

    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS: The CUFFT operation was performed";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN: The CUFFT plan to execute is invalid";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED: The allocation of data for CUFFT in memory failed";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE: The data type used by CUFFT is invalid";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE: The data value used by CUFFT is invalid";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR: An internal error occurred in CUFFT";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED: The execution of a plan by CUFFT failed";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED: The setup of CUFFT failed";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE: The size of the data to be used by CUFFT is invalid";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA: The data to be used by CUFFT is unaligned in memory";
    }

    return "Unknown CUFFT Error";
}

void compute_fft_2D_R2C(blitz::Array<float, 2> &grid, void *data)
{
    int n[] = {grid.rows(), grid.cols()}; // 2D FFT of length NxN
    int inembed[] = {grid.rows(), 2 * (grid.cols() / 2 + 1)};
    int onembed[] = {grid.rows(), (grid.cols() / 2 + 1)};
    int howmany = 1;
    int odist = grid.rows() * (grid.cols() / 2 + 1); // Output distance is in "complex"
    int idist = 2 * odist;                           // Input distance is in "real"
    int istride = 1;                                 // Elements of each FFT are adjacent
    int ostride = 1;

    cufftHandle plan;
/*
    cufftPlanMany(&plan, sizeof(n) / sizeof(n[0]), n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_R2C, howmany);
*/


    size_t workSize;
    void* workArea;
    cufftCreate(&plan);
    cufftSetAutoAllocation(plan,0);
    cufftMakePlanMany(plan,sizeof(n) / sizeof(n[0]), n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_R2C, howmany, &workSize);
   
    auto data_size = sizeof(cufftComplex) * howmany * grid.rows() * (grid.cols() / 2 + 1);
    cudaMalloc((void**)&workArea, workSize);
    cufftSetWorkArea(plan, workArea);
    
    cudaMemcpy(data, grid.dataFirst(), data_size, cudaMemcpyHostToDevice);

    auto status = cufftExecR2C(plan, reinterpret_cast<cufftReal *>(data), reinterpret_cast<cufftComplex *>(data));
    if (status != CUFFT_SUCCESS)
    {
        const char *errorString = cufftGetErrorString(status);
        printf("CUDA cuFFT Error: %s\n", errorString);
    }

    cudaMemcpy(grid.dataFirst(), data, data_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //    cudaFree(data);
    cufftDestroy(plan);
}

void compute_fft_2D_R2C_Stream(blitz::Array<float, 2> &grid, void *data, cufftHandle* plan, void* workArea, cudaStream_t* stream)
{
    int howmany = 1;
    //    cufftComplex *data;
    auto data_size = sizeof(cufftComplex) * howmany * grid.rows() * (grid.cols() / 2 + 1);
    //    cudaMalloc((void**)&data, data_size);
    cudaMemcpyAsync(data, grid.dataFirst(), data_size, cudaMemcpyHostToDevice, *stream);
    cufftSetStream(*plan, *stream);
    cufftSetWorkArea(*plan, workArea);
    auto status = cufftExecR2C(*plan, reinterpret_cast<cufftReal *>(data), reinterpret_cast<cufftComplex *>(data));
    if (status != CUFFT_SUCCESS)
    {
        const char *errorString = cufftGetErrorString(status);
        printf("CUDA cuFFT Error: %s\n", errorString);
    }

    cudaMemcpyAsync(grid.dataFirst(), data, data_size, cudaMemcpyDeviceToHost, *stream);
    cudaDeviceSynchronize();
    //    cudaFree(data);
}

void *allocate_cuda_slab(size_t nGrid)
{
    void *cuda_slab;
    auto slab_size = sizeof(cufftComplex) * nGrid * (nGrid / 2 + 1);
    cudaMalloc((void **)&cuda_slab, slab_size);
    return cuda_slab;
}
void* allocate_cuda_slab_Stream(size_t nGrid, cudaStream_t* stream) {
    void* cuda_slab;
    auto slab_size = sizeof(cufftComplex) * nGrid * (nGrid / 2 + 1);
    cudaMallocAsync((void**)&cuda_slab, slab_size, *stream);
    return cuda_slab;
}

void* allocate_cuda_size(size_t size) {
    void* cuda_data;
    cudaMalloc((void **)&cuda_data, size);
    return cuda_data;
}

void* allocate_cuda_size_Stream(size_t size, cudaStream_t* stream) {
    void* cuda_data;
    cudaMallocAsync((void **)&cuda_data, size, *stream);
    return cuda_data;
}

void destroy_cuda_data(void* data) {
    cudaFree(data);
}

void destroy_cuda_data_Stream(void* data, cudaStream_t* stream) {
    cudaFreeAsync(data, *stream);
}
