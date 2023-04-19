//
//  main.cpp
//  HPC
//
//  Created by yang qian on 2023/3/4.
//

// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <complex>
#include <new>
#include <blitz/array.h>
#include "omp.h"
#include "tipsy.hpp"
#include "my_weights.h"
#include <fftw3.h>

typedef std::chrono::system_clock::time_point timePointType;
typedef std::chrono::duration<double> durationType;
typedef float coord_Type[3];
typedef typename blitz::Array<float, 3> M3fType;
typedef typename blitz::Array<float, 2> M2fType;
typedef typename blitz::Array<std::complex<float>, 3> M3cType;

void NGP(float, float, float, int, M3fType&);
void CIC(float, float, float, int, M3fType&);
void TSC(float, float, float, int, M3fType&);
void PCS(float, float, float, int, M3fType&);
void outPut(const char*, int, const M2fType&);
void func();
float PCS_return(float, float, float, int, M3fType&);

void LinearBinning(int, const M3cType&);
void VariableBinning(int, int, const M3cType&);
void LogBinning(int, int, const M3cType&);

int main(int argc, char *argv[]) {
    
    if (argc<=1) {
            std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size]"
                      << std::endl;
            return 1;
        }

    int nGrid = 100;
    int massOption;
    assert(argc > 3);
    nGrid = atoi(argv[2]);
    massOption = atoi(argv[3]);
    void (*assignMass) (float, float, float, int, M3fType&);
    if (massOption == 0) {
        printf("Nearest Grid Point.\n");
        assignMass = &NGP;
    }
    else if (massOption == 1) {
        printf("Cloud in Cell.\n");
        assignMass = &CIC;
    }
    else if(massOption == 2) {
        printf("Triangle Shaped Cloud.\n");
        assignMass = &TSC;
    }
    else if (massOption == 3) {
        printf("Piecewise Cubic Spline.\n");
        assignMass = &PCS;
    }
    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }

    if (unit_Test()) printf("Test passed.\n");
    std::uint64_t N = io.count();

    // Load particle positions
    std::cerr << "Loading " << N << " particles" << std::endl;
    
    M2fType r(N, 3);
    timePointType start = std::chrono::system_clock::now();
    io.load(r);
    timePointType end = std::chrono::system_clock::now();
    durationType duration = end - start;
    printf("Reading file took: %.8f s\n", duration);
    
    float* datawPadding = new (std::align_val_t(64)) float[nGrid * nGrid * (nGrid + 2)];
    M3fType gridwPadding(datawPadding, blitz::shape(nGrid, nGrid, nGrid+2), blitz::neverDeleteData);
    gridwPadding = 0.0;
    M3fType grid = gridwPadding(blitz::Range::all(), blitz::Range::all(), blitz::Range(0, nGrid-1));
    
    std::complex<float>* dataComplex = reinterpret_cast<std::complex<float>*>(datawPadding);
    M3cType kGrid(dataComplex, blitz::shape(nGrid, nGrid, nGrid / 2 + 1));
    start = std::chrono::system_clock::now();

#pragma omp parallel
{
#pragma omp for
    for (int pn = 0; pn < N; pn++) {
        float x = (r(pn, 0) + 0.5);
        float y = (r(pn, 1) + 0.5);
        float z = (r(pn, 2) + 0.5);
        if (abs(x-1.0) < 0.0001) x -= 0.0001;
        if (abs(y-1.0) < 0.0001) y -= 0.0001;
        if (abs(z-1.0) < 0.0001) z -= 0.0001;
	x *= nGrid;
	y *= nGrid;
	z *= nGrid;
        assert(x >= 0 && x < nGrid);
        assert(y >= 0 && y < nGrid);
        assert(z >= 0 && z < nGrid);
        assignMass(x, y, z, nGrid, grid);
    }
    
}
    
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("Mass assignment took: %.8f s\n", duration);
    
    float blitzSum = blitz::sum(grid);
    printf("totalSum = %f, ", blitzSum);
    float average_density = blitzSum / (nGrid * nGrid * nGrid);
    printf("average_density = %f\n", average_density);
    grid -= average_density;
    grid /= average_density;
    printf("overall overdensity = %f\n", blitz::sum(grid));
   /*
    M2fType projected(nGrid, nGrid);
    projected = 0.0;
    
    start = std::chrono::system_clock::now();
    blitz::thirdIndex k;
    projected = blitz::max(grid, k);
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("Projection took: %.8f s\n", duration);
    */

    start = std::chrono::system_clock::now();
    fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid, nGrid, nGrid, datawPadding, (fftwf_complex *)dataComplex, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("FFT took: %.8f s\n", duration);
    LinearBinning(nGrid, kGrid);
    VariableBinning(nGrid, nGrid * 0.8, kGrid);
    LogBinning(nGrid, nGrid * 0.8, kGrid);
    delete [] datawPadding;
    return 0;
}

void func() {
#ifdef _OPENMP
    bool p = omp_in_parallel();
    printf("%s\n", p ? "parallel" : "serial");
#endif
}

void NGP(float x, float y, float z, int nGrid, M3fType& grid) {
#pragma omp parallel
{
    int i = std::floor(x);
    int j = std::floor(y);
    int k = std::floor(z);
#pragma omp atomic
    grid(i, j, k) += 1.0;
}
}

void CIC(float x, float y, float z, int nGrid, M3fType& grid) {
#pragma omp parallel
{
    float Wx[2];
    float Wy[2];
    float Wz[2];
    int ix = cic_weights(x, Wx);
    int iy = cic_weights(y, Wy);
    int iz = cic_weights(z, Wz);
    float totalW = 0.0;
#pragma omp for collapse(3)
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int indexI = (ix + i + nGrid) % nGrid;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI < nGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
                grid(indexI, indexJ, indexK) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.0001);
}
    
}

void TSC(float x, float y, float z, int nGrid, M3fType& grid) {
#pragma omp parallel
{
    float Wx[3];
    float Wy[3];
    float Wz[3];
    int ix = tsc_weights(x, Wx);
    int iy = tsc_weights(y, Wy);
    int iz = tsc_weights(z, Wz);
    float totalW = 0.0;
#pragma omp for collapse(3)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int indexI = (ix + i + nGrid) % nGrid;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI < nGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
                grid(indexI, indexJ, indexK) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.0001);
}
    
}

void PCS(float x, float y, float z, int nGrid, M3fType& grid) {
#pragma omp parallel
{
    float Wx[4];
    float Wy[4];
    float Wz[4];
    int ix = pcs_weights(x, Wx);
    int iy = pcs_weights(y, Wy);
    int iz = pcs_weights(z, Wz);
    float totalW = 0.0;
#pragma omp for collapse(3)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                int indexI = (ix + i + nGrid) % nGrid;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI < nGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
                grid(indexI, indexJ, indexK) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.00001);
}
    
}

float PCS_return(float x, float y, float z, int nGrid, M3fType& grid) {
    float Wx[4];
    float Wy[4];
    float Wz[4];
    int ix = pcs_weights(x, Wx);
    int iy = pcs_weights(y, Wy);
    int iz = pcs_weights(z, Wz);
    float totalW = 0.0;
#pragma omp parallel for collapse(3)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                int indexI = (ix + i + nGrid) % nGrid;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI < nGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
                grid(indexI, indexJ, indexK) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.0001);
    return totalW;
}


void outPut(const char* fileName, int nGrid, const M2fType& projected) {
    std::ofstream fout(fileName);
    if (fout.is_open()) {
        for (int i = 0; i < nGrid; i++) {
            for (int j = 0; j < nGrid; j++) {
                fout << projected(i, j) << ",";
            }
            fout << std::endl;
        }
    }
    fout.close();
}

void LinearBinning(int nGrid, const M3cType& kGrid) {
    blitz::Array<unsigned int, 1> nPower(nGrid);
    blitz::Array<double, 1> fPower(nGrid);
    nPower = 0;
    fPower = 0.0;

    for (int i = 0; i < nGrid; i++) {
            int kx = (i <= nGrid/2) ? i : -nGrid + i;
            for (int j = 0; j < nGrid; j++) {
                    int ky = (j <= nGrid/2) ? j : -nGrid + j;
                    for (int k = 0; k < (nGrid / 2 + 1); k++) {
                            int kz = k;
                            double kf = std::sqrt(kx * kx + ky * ky + kz * kz);
                            int kIndex = std::floor(kf);
                            assert(kIndex < nGrid);
                            nPower(kIndex)++;
                            fPower(kIndex) += std::norm(kGrid(i, j, k));
                        }
            }

    }
    for (int i = 0; i < nGrid; i++) {
            if (nPower(i)) fPower(i) /= nPower(i);
    }

    std::string fileName = "Linear_";
    fileName += std::to_string(nGrid);
    fileName += ".txt";
    std::ofstream fout(fileName.c_str());
    if (fout.is_open()) {
            for (int i = 0; i < nGrid; i++) {
                    fout << fPower(i) << std::endl;
                }
    }

}

void VariableBinning(int nGrid, int nBins, const M3cType& kGrid) {
    blitz::Array<unsigned long, 1> nPower(nBins);
    blitz::Array<double, 1> fPower(nBins);
    nPower = 0;
    fPower = 0.0;
    int maxBin = std::floor(std::sqrt(3) * nGrid / 2.0);
    printf("maxBin = %d\n", maxBin);

    for (int i = 0; i < nGrid; i++) {
            int kx = (i <= nGrid/2) ? i : -nGrid + i;
            for (int j = 0; j < nGrid; j++) {
                    int ky = (j <= nGrid/2) ? j : -nGrid + j;
                    for (int k = 0; k < (nGrid / 2 + 1); k++) {
                            int kz = k;
                            double kf = std::sqrt(kx * kx + ky * ky + kz * kz);
                            int kIndex = std::floor(kf / maxBin * nBins);
                            if (kf / maxBin >= 1) kIndex--;
			   
			    assert(kIndex < nBins);
                            nPower(kIndex)++;
                            fPower(kIndex) += std::norm(kGrid(i, j, k));
			    //printf("kf:%f, kIndex:%d, power:%f\n",kf, kIndex, std::norm(kGrid(i, j, k)));
                        }
            }

    }
    for (int i = 0; i < nBins; i++) {
            if (nPower(i)) fPower(i) /= nPower(i);
    }

    std::string fileName = "Variable";
    fileName += "_";
    fileName += std::to_string(nGrid);
    fileName += "_";
    fileName += std::to_string(nBins);
    fileName += ".txt";
    std::ofstream fout(fileName.c_str());
    if (fout.is_open()) {
            for (int i = 0; i < nBins; i++) {
                    fout << fPower(i) << std::endl;
                }
    }
}

void LogBinning(int nGrid, int nBins, const M3cType& kGrid) {
    blitz::Array<unsigned long, 1> nPower(nBins);
    blitz::Array<double, 1> fPower(nBins);
    nPower = 0;
    fPower = 0.0;
    int maxBin = std::floor(std::sqrt(3) * nGrid / 2.0);
    printf("maxBin = %d\n", maxBin);

    for (int i = 0; i < nGrid; i++) {
            int kx = (i <= nGrid/2) ? i : -nGrid + i;
            for (int j = 0; j < nGrid; j++) {
                    int ky = (j <= nGrid/2) ? j : -nGrid + j;
                    for (int k = 0; k < (nGrid / 2 + 1); k++) {
                            int kz = k;
			    if (kx == 0 && ky == 0 && kz == 0) continue;
                            double kf = std::sqrt(kx * kx + ky * ky + kz * kz);
                            int kIndex = std::floor(std::log(kf) / std::log(maxBin) * nBins);
                            if (kf / maxBin >= 1) kIndex--;

                            assert(kIndex < nBins);
                            nPower(kIndex)++;
                            fPower(kIndex) += std::norm(kGrid(i, j, k));
                           //printf("kf:%f, kIndex:%d, power:%f\n",kf, kIndex, std::norm(kGrid(i, j, k)));
                        }
            }

    }

    for (int i = 0; i < nBins; i++) {
            if (nPower(i)) fPower(i) /= nPower(i);
    }

    std::string fileName = "Log";
    fileName += "_";
    fileName += std::to_string(nGrid);
    fileName += "_";
    fileName += std::to_string(nBins);
    fileName += ".txt";
    std::ofstream fout(fileName.c_str());
    if (fout.is_open()) {
            for (int i = 0; i < nBins; i++) {
                    fout << fPower(i) << std::endl;
                }
    }




}
