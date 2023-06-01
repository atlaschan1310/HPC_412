CXX		= CC
CXXFLAGS	= -O3 -g -march=native
OMP		= -fopenmp
INC_DIR=/users/yqian/blitz/usr/local/include
CUDA_DIR=/store/uzh/uzh8/packages/include
assign	: assign.o tipsy.o gpufft.o
	$(CXX) $(OMP) -o assign assign.o gpufft.o tipsy.o -lfftw3f -lfftw3f_mpi -lcufft

assign.o : assign.cxx tipsy.h gpufft.h
	$(CXX) $(OMP) $(CXXFLAGS) -O3 -std=c++17 -c -o assign.o assign.cxx -I$(INC_DIR) -I$(CUDA_DIR)

gpufft.o : gpufft.cu gpufft.h
	nvcc  -O3 -std=c++17 -c -o gpufft.o gpufft.cu -I$(INC_DIR) -I$(CUDA_DIR)

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) $(CXXFLAGS) -O3 -std=c++17 -c -o tipsy.o tipsy.cxx -I$(INC_DIR) -I$(CUDA_DIR)


clean:
	rm -f assign assign.o tipsy.o gpufft.o fft10-cpu fft10-gpu
