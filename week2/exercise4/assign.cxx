// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include "blitz/array.h"
#include "tipsy.h"
using namespace blitz;

int main(int argc, char *argv[]) {
    if (argc<=1) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc>2) nGrid = atoi(argv[2]);

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    std::uint64_t N = io.count();

    // Load particle positions
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float,2> r(N,3);
    io.load(r);

    // Create Mass Assignment Grid
    Array<float,3> grid(nGrid,nGrid,nGrid);

    grid = 0;
    for(int pn=0; pn<N; ++pn) {
	float x = r(pn,0);
	float y = r(pn,1);
	float z = r(pn,2);
    x += 0.5;
    y += 0.5;
    z += 0.5;
    int i = x * nGrid;
    int j = y * nGrid;
    int k = z * nGrid;
    grid(i,j,k)++;

	// Convert x, y and z into a grid position i,j,k such that
	// 0 <= i < nGrid
	// 0 <= j < nGrid
	// 0 <= k < nGrid

	// Deposit the mass onto grid(i,j,k)
    }
    Array<float, 2> projected(nGrid, nGrid);
    thirdIndex k;
    projected = blitz::max(grid, k);
    std::ofstream fout("projectout.txt");
    if (fout.is_open()) {
        for (int i = 0; i < nGrid; i++) {
            for (int j = 0; j < nGrid; j++) {
                fout << projected(i, j) << ",";
            }
            fout << "\n";
        }
    }
    fout.close();
    return 0;
}

