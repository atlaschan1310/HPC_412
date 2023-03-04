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
#include "blitz/array.h"

using namespace blitz;

int main(int argc, char *argv[]) {
    GeneralArrayStorage<3> storage;
    storage.base() = 10, 0, 0;
    storage.ordering() = firstDim, secondDim, thirdDim;
    Array<int, 3> A(Range(10, 14), Range(0, 19), Range(0, 19), storage);
    std::cout << A << std::endl;
    return 0;
}

