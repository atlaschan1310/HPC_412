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
#include <new>
#include <blitz/array.h>
#include "tipsy.hpp"
#include "omp.h"
typedef std::chrono::system_clock::time_point timePointType;
typedef std::chrono::duration<double> durationType;
typedef float coord_Type[3];
typedef typename blitz::Array<float, 3> M3fType;
typedef typename blitz::Array<float, 2> M2fType;

int sign(float);
float calS(int*, float, float, int);
void NGP(float, float, float, int, float, M3fType&);
void CIC(float, float, float, int, float, M3fType&);
void TSC(float, float, float, int, float, M3fType&);
void PCS(float, float, float, int, float, M3fType&);
void outPut(const char*, int, const M2fType&);
void func();

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
    assert(massOption >= 0 && massOption < 4);
    void (*assignMass) (float, float, float, int, float, M3fType&);
    if (massOption == 0) {
        assignMass = &NGP;
    }
    else if (massOption == 1) {
        assignMass = &CIC;
    }
    else if(massOption == 2) {
        assignMass = &TSC;
    }
    else if (massOption == 3) {
        assignMass = &PCS;
    }
    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    
    std::uint64_t N = io.count();

    // Load particle positions
    std::cerr << "Loading " << N << " particles" << std::endl;

    M2fType r(N, 3);
    timePointType start = std::chrono::system_clock::now();
    io.load(r);
    timePointType end = std::chrono::system_clock::now();
    durationType duration = end - start;
    printf("Reading file took: %.8f s\n", duration);
    M3fType grid(nGrid, nGrid, nGrid);
    grid = 0.0;
    
    start = std::chrono::system_clock::now();
#pragma omp parallel
{
#pragma omp for
    for (int pn = 0; pn < N; pn++) {
        float x = r(pn, 0) + 0.5;
        float y = r(pn, 1) + 0.5;
        float z = r(pn, 2) + 0.5;
    if (abs(x-1.0) < 0.0001) x -= 0.0001;
    if (abs(y-1.0) < 0.0001) y -= 0.0001;
    if (abs(z-1.0) < 0.0001) z -= 0.0001;
    assignMass(x, y, z, nGrid, 1.0/nGrid, grid);
    }
}
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("Mass assignment took: %.8f s\n", duration);
    
    M2fType projected(nGrid, nGrid);
    projected = 0.0;
    
    start = std::chrono::system_clock::now();
    blitz::thirdIndex k;
    projected = blitz::max(grid, k);
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("Projection took: %.8f s\n", duration);
   /*
    char* f1 = "out";
    char* f2 = argv[2];
    char* f3 = ".csv";
    int length = strlen(f1) + strlen(f2) + strlen(f3);
    char* res = new char[length+1]();
    strncpy(res, f1, strlen(f1));
    strncpy(res + strlen(f1), f2, strlen(f2));
    strcpy(res + strlen(f1) + strlen(f2), f3);
    std::cout << res << std::endl;
    outPut(res, nGrid, projected);
    delete [] res;
   */
        
    return 0;
}

void func() {
#ifdef _OPENMP
    bool p = omp_in_parallel();
    printf("%s\n", p ? "parallel" : "serial");
#endif
}


float calS(int* index, float coord, float gridLength, int nGrid) {
    float res = (coord - (*index * gridLength + 0.5 * gridLength)) / gridLength;
    if (*index < 0) {
        *index = *index + nGrid;
    }
    else if (*index >= nGrid) {
        *index = *index - nGrid;
    }
    assert(*index >= 0 && *index < nGrid);
    return res;
}

int sign(float x) {return (x >= 0) ? 1 : -1;}

void NGP(float x, float y, float z, int nGrid, float gridLength, M3fType& grid) {
#pragma omp parallel
{
    int i = x * nGrid;
    int j = y * nGrid;
    int k = z * nGrid;
#pragma omp atomic
    grid(i, j, k) += 1.0;
}
}

void CIC(float x, float y, float z, int nGrid, float gridLength, M3fType& grid) {
#pragma omp parallel
{
    int i = x * nGrid;
    int j = y * nGrid;
    int k = z * nGrid;
    float sX = calS(&i, x, gridLength, nGrid);
    float sY = calS(&j, y, gridLength, nGrid);
    float sZ = calS(&k, z, gridLength, nGrid);
    int offset[8][3] = {
        {0,0,0},{0,sign(sY),0},{0,0,sign(sZ)},{0,sign(sY),sign(sZ)},
        {sign(sX),0,0},{sign(sX),sign(sY),0},{sign(sX),0,sign(sZ)},{sign(sX),sign(sY),sign(sZ)}
    };
    float totalW = 0.0;
    for (int index = 0; index < 8; index++) {
        int neighborI = i + offset[index][0];
        int neighborJ = j + offset[index][1];
        int neighborK = k + offset[index][2];
        float curSX = calS(&neighborI, x, gridLength, nGrid);
        float curSY = calS(&neighborJ, y, gridLength, nGrid);
        float curSZ = calS(&neighborK, z, gridLength, nGrid);
        float wX = std::max(0.0, 1.0 - abs(curSX));
        float wY = std::max(0.0, 1.0 - abs(curSY));
        float wZ = std::max(0.0, 1.0 - abs(curSZ));
        float W = wX * wY * wZ;
#pragma omp atomic
        grid(i, j, k) += W;
        totalW += W;
    }
    assert(abs(totalW - 1.0) < 0.0001);
}
    
}

void TSC(float x, float y, float z, int nGrid, float gridLength, M3fType& grid) {
    auto tsc = [](float s) -> float {
        float res = 0.0;
        if (s < 0.5) {
            res = 0.75 - s * s;
        }
        else if (s >= 0.5 && s < 1.5) {
            res = 0.5 * (1.5 - s) * (1.5 - s);
        }
        return res;
    };
#pragma omp parallel
{
    int i = x * nGrid;
    int j = y * nGrid;
    int k = z * nGrid;
    
    float totalW = 0.0;
    int offset[3] = {0, -1, 1};
    for (int xOff = 0; xOff < 3; xOff++) {
        for (int yOff = 0; yOff < 3; yOff++) {
            for (int zOff = 0; zOff < 3; zOff++) {
                int neighborI = i + offset[xOff];
                int neighborJ = j + offset[yOff];
                int neighborK = k + offset[zOff];
                float curSX = calS(&neighborI, x, gridLength, nGrid);
                float curSY = calS(&neighborJ, y, gridLength, nGrid);
                float curSZ = calS(&neighborK, z, gridLength, nGrid);
                float W = tsc(abs(curSX)) * tsc(abs(curSY)) * tsc(abs(curSZ));
#pragma omp atomic
               grid(i, j, k) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.0001);
}
    
}

void PCS(float x, float y, float z, int nGrid, float gridLength, M3fType& grid) {
    auto pcs = [](float s) -> float {
        float res = 0.0;
        if (s < 1) {
            res = (4 - 6 * s * s + 3 * s * s * s) / 6.0;
        }
        else if (s < 2) {
            res = ((2 - s) * (2 - s) * (2 - s)) / 6.0;
        }
        return res;
    };
#pragma omp parallel
{
    int i = x * nGrid;
    int j = y * nGrid;
    int k = z * nGrid;
    float sX = calS(&i, x, gridLength, nGrid);
    float sY = calS(&j, y, gridLength, nGrid);
    float sZ = calS(&k, z, gridLength, nGrid);
    int offsetX[4] = {0, 1, -1, 2 * sign(sX)};
    int offsetY[4] = {0, 1, -1, 2 * sign(sY)};
    int offsetZ[4] = {0, 1, -1, 2 * sign(sZ)};
    float totalW = 0.0;
    for (int xOff = 0; xOff < 4; xOff++) {
        for (int yOff = 0; yOff < 4; yOff++) {
            for (int zOff = 0; zOff < 4; zOff++) {
                int neighborI = i + offsetX[xOff];
                int neighborJ = j + offsetY[yOff];
                int neighborK = k + offsetZ[zOff];
                float curSX = calS(&neighborI, x, gridLength, nGrid);
                float curSY = calS(&neighborJ, y, gridLength, nGrid);
                float curSZ = calS(&neighborK, z, gridLength, nGrid);
                float W = pcs(abs(curSX)) * pcs(abs(curSY)) * pcs(abs(curSZ));
#pragma omp atomic
               grid(i, j, k) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.0001);
}
    
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

