//
//  main.cpp
//
//  Created by yang qian on 2022/10/20.
//

#include "blitz/array.h"
#include <iostream>

template<typename T, int N>
class myArray {
private:
    blitz::Array<T,N> data;
public:
    myArray(blitz::Array<T, N>& _data) : data(_data) {};
    ~myArray() {};
    friend std::ostream& operator<<(std::ostream& os, const myArray<T, N>& data) {
        os << "data" << std::endl;
        return os;
    };
};

int main(int argc, const char * argv[]) {
    blitz::Array<float, 2> data(2,2);
    myArray<float, 2> array(data);
    std::cout << array << std::endl;
    std::cout << data << std::endl;
    std::cout << "data" << std::endl;
    return 0;
}

