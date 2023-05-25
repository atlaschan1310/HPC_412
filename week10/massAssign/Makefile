CXX	= nvcc -lcufft 
LDFLAGS= -lcufft
INC_DIR=/users/yqian/blitz/usr/local/include
CUDA_DIR=/store/uzh/uzh8/packages/include


assign	: assign.o tipsy.o
	$(CXX) -O3 -o assign assign.o tipsy.o

assign.o : assign.cu tipsy.hpp my_weights.h
	nvcc -O3 -I$(INC_DIR) -I$(CUDA_DIR) -lcufft -std=c++17 -c -o assign.o assign.cu

tipsy.o : tipsy.cpp tipsy.hpp	
	$(CXX) -O3 -I$(INC_DIR) -std=c++17 -c -o tipsy.o tipsy.cpp


clean:
	rm -f assign assign.o tipsy.o

