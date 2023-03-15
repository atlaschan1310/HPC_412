CXX	= g++ -fopenmp
LDFLAGS=-fopenmp

assign	: assign.o tipsy.o
	$(CXX) -O3 -o assign assign.o tipsy.o

assign.o : assign.cpp tipsy.hpp
	$(CXX) -O3 -std=c++17 -c -o assign.o assign.cpp

tipsy.o : tipsy.cpp tipsy.hpp	
	$(CXX) -O3 -std=c++17 -c -o tipsy.o tipsy.cpp

clean:
	rm -f assign assign.o tipsy.o

