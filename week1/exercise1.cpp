//
//  main.cpp
//
//  Created by yang qian on 2022/10/20.
//

#include "blitz/array.h"
#include <iostream>

typedef typename blitz::Array<float, 3> data_Type;

std::ostream& operator<<(std::ostream& os, const data_Type& data) {
    os << "3-Dimensional Data" << std::endl;
    os << data.ubound(0) + 1 << "x" << data.ubound(1) + 1 << "x" << data.ubound(2) + 1 << std::endl;
    for (int i = data.lbound(2); i <= data.ubound(2); i++) {
        for (int j = data.lbound(0); j <= data.ubound(0); j++) {
            for (int k = data.lbound(1); k <= data.ubound(1); k++) {
                os << data(j,k,i);
                if (k < data.ubound(1)) os << ",";
            }
            os << '\n';
        }
        os << "-------------" << std::endl;
    }
    return os;
}

int main(int argc, const char * argv[]) {
    data_Type data(10,8,6);
    std::cout << data << std::endl;
    return 0;
}


