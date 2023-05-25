#include <blitz/array.h>
#include <complex>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <cufftw.h>
using namespace blitz;
using std::complex;

void fill_array(Array<float,2> &data) {
    // Set the grid to the sum of two sine functions
    for (int i=0; i < data.rows(); i++) {
        for (int j=0; j < data.cols(); j++) {
            float x = (float)i / 25.0; // Period of 1/4 of the box in x
            float y = (float)j / 10.0; // Period of 1/10 of the box in y
            data(i,j) = sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y);
        }
    }
}

// Verify the FFT (kdata) of data by performing a reverse transform and comparing
bool validate(Array<float,2> &data,Array<std::complex<float>, 2> kdata) {
    Array<float,2> rdata(data.extent());
    fftwf_plan plan = fftwf_plan_dft_c2r_2d(data.rows(), data.cols(),
        reinterpret_cast<fftwf_complex*>(kdata.data()), rdata.data(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    rdata /= data.size(); // Normalize for the FFT
    return all(abs(data - rdata) < 1e-5);
}
template<typename T>
bool validate2d(const Array<T,2>& r1, const Array<T,2>& r2) {
	return all(abs(r1-r2) < 1e-5);
}

int main() {
    int n = 10000;
   /*
    // Out of place
    Array<float,2> rdata1(n,n);
    Array<std::complex<float>, 2> kdata1(n, n/2 + 1);
    fftwf_plan plan1  = fftwf_plan_dft_r2c_2d(n, n,
        rdata1.data(), reinterpret_cast<fftwf_complex*>(kdata1.data()), FFTW_ESTIMATE);
    fill_array(rdata1);
    fftwf_execute(plan1);
    fftwf_destroy_plan(plan1);
    std::cout << ">>> Out of place FFT " << (validate(rdata1,kdata1)?"match":"MISMATCH") << endl;

    // in-place
    Array<float,2> raw_data2(n,n+2);
    Array<float,2> rdata2 = raw_data2(Range(0,n-1),Range(0,n-1));
    fftwf_plan plan2  = fftwf_plan_dft_r2c_2d(n, n,
        rdata2.data(), reinterpret_cast<fftwf_complex*>(rdata2.data()), FFTW_ESTIMATE);
    fill_array(rdata2);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
    Array<std::complex<float>, 2> kdata2(reinterpret_cast<std::complex<float>*>(rdata2.data()),
        shape(n, n/2 + 1),neverDeleteData);
    std::cout << ">>> In-place FFT " << (validate(rdata1,kdata2)?"match":"MISMATCH") << endl;
    //Exercise3   
    Array<float,2> raw_data3(n,n+2);
    Array<float,2> data3 = raw_data3(blitz::Range::all(), blitz::Range(0,n-1));
    fill_array(data3);
    Array<float,2> raw_data4(n,n+2);
    void* device_data;
    size_t size_in_byte = sizeof(float) * n * (n + 2);
    cudaMalloc(&device_data, size_in_byte);
    cudaMemcpy(device_data, raw_data3.data(), size_in_byte, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaMemcpy(raw_data4.data(), device_data, size_in_byte, cudaMemcpyDeviceToHost);
    std::cout << (validate2d(raw_data3, raw_data4) ? "passed" : "failed") << std::endl;
    cudaFree(device_data);
    */
    Array<float,2> raw_data5(n,n+2);
    Array<float,2> data5 = raw_data5(blitz::Range::all(), blitz::Range(0,n-1));
    Array<std::complex<float>,2> data6(n, n/2+1);
    float* device_data;
    size_t size_in_byte = sizeof(float) * n * (n + 2);
    cudaMalloc(&device_data, size_in_byte);
    cudaMemcpy(device_data, raw_data5.data(), size_in_byte, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cufftHandle plan;
    int dim[2] = {n,n};
    int inembed[2] = {n,n+2};
    int onembed[2] = {n, n/2+1};
    int batch = 1;
    int odist = 1;
    int idist = 2;
    int istride = 1;
    int ostride = 1;

    cufftPlanMany(&plan, 2, dim, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
    cufftExecR2C(plan, device_data, (cufftComplex*)device_data);
    cudaDeviceSynchronize();
    cufftDestroy(plan);
    cudaMemcpy(data6.data(), device_data, size_in_byte, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(device_data);  
    std::cout << ">>> cuda FFT " << (validate(data5,data6)?"match":"MISMATCH") << endl;
    return 0;
 }
