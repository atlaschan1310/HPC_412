CXX	= CC -fopenmp
LDFLAGS=-fopenmp -lfftw3 -lm -lfftw3f -lfftw3_mpi
INC_DIR= /users/yqian/blitz/usr/local/include

assign	: assign.o tipsy.o
	$(CXX) -O3 -o assign assign.o tipsy.o

assign.o : assign.cxx tipsy.h weights.h
	$(CXX) -O3 -I$(INC_DIR) -std=c++17 -c -o assign.o assign.cxx

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) -O3 -I$(INC_DIR) -std=c++17 -c -o tipsy.o tipsy.cxx

clean:
	rm -f assign assign.o tipsy.o

