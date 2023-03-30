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

typedef std::chrono::system_clock::time_point timePointType;
typedef std::chrono::duration<double> durationType;
typedef float coord_Type[3];
typedef typename blitz::Array<float, 3> M3fType;
typedef typename blitz::Array<float, 2> M2fType;

void NGP(float, float, float, int, M3fType&);
void CIC(float, float, float, int, M3fType&);
void TSC(float, float, float, int, M3fType&);
void PCS(float, float, float, int, M3fType&);
void outPut(const char*, int, const M2fType&);
void func();
float PCS_return(float, float, float, int, M3fType&);

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
    
    
    /*
    int nGrid = 100;
    int massOption = 3;
    const char* fileName = "/Users/yangqian/Documents/UZHCS/AdvancedHighPerformanceComputing/B100.00100";
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
    */
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
    
    float* data = new (std::align_val_t(64)) float[nGrid * nGrid * nGrid];
    M3fType grid(data, blitz::shape(nGrid, nGrid, nGrid), blitz::neverDeleteData);
    grid = 0.0;
    
    start = std::chrono::system_clock::now();
   //double total = 0.0;
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
/*
#pragma omp atomic 
	total += PCS_return(x, y, z, nGrid, grid);
*/
    }
    
}
    
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("Mass assignment took: %.8f s\n", duration);
    
    float blitzSum = blitz::sum(grid);
    printf("totalSum = %f\n", blitzSum);
    assert(blitzSum == (nGrid * nGrid * nGrid));
    
    M2fType projected(nGrid, nGrid);
    projected = 0.0;
    
    start = std::chrono::system_clock::now();
    blitz::thirdIndex k;
    projected = blitz::max(grid, k);
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("Projection took: %.8f s\n", duration);
    delete [] data;
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


