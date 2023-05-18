#include <string>
#include "blitz/array.h"

int main(int argc, char *argv[]);

template <typename T> void write(std::string location, blitz::Array<T,2> data);
