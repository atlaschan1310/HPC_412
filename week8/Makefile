CXX	= CC -fopenmp
LDFLAGS=-fopenmp -lfftw3 -lm -lfftw3f -lfftw3_mpi
INC_DIR= /users/yqian/blitz/usr/local/include

assign	: assign.o tipsy.o
	$(CXX) -O3 -o assign assign.o tipsy.o

assign.o : assign.cpp tipsy.hpp my_weights.h
	$(CXX) -O3 -I$(INC_DIR) -std=c++17 -c -o assign.o assign.cpp

tipsy.o : tipsy.cpp tipsy.hpp	
	$(CXX) -O3 -I$(INC_DIR) -std=c++17 -c -o tipsy.o tipsy.cpp

clean:
	rm -f assign assign.o tipsy.o

