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
#include <math.h>
#include <chrono>
#include <complex>
#include <new>
#include <blitz/array.h>
#include "omp.h"
#include "tipsy.hpp"
#include "my_weights.h"
#include <fftw3.h>
#include <mpi.h>
#include <fftw3-mpi.h>

typedef std::chrono::system_clock::time_point timePointType;
typedef std::chrono::duration<double> durationType;
typedef float coord_Type[3];
typedef typename blitz::Array<float, 3> M3fType;
typedef typename blitz::Array<float, 2> M2fType;
typedef typename blitz::Array<std::complex<float>, 3> M3cType;

void NGP(float, float, float, int, int, M3fType&);
void CIC(float, float, float, int, int, M3fType&);
void TSC(float, float, float, int, int, M3fType&);
void PCS(float, float, float, int, int, M3fType&);
void outPut(const char*, int, const M2fType&);
void func();
float PCS_return(float, float, float, int, int, M3fType&);

void swapCoord(coord_Type*, int, int);
int partition(coord_Type*, int, int, int, int);

void LinearBinning(int, const M3cType&);
void VariableBinning(int, int, const M3cType&);
void LogBinning(int, int, const M3cType&);

int main(int argc, char *argv[]) {
    
    if (argc<=1) {
            std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size]"
                      << std::endl;
            return 1;
        }
    assert(argc > 3);
    int massOption;
    int nGrid = atoi(argv[2]);
    massOption = atoi(argv[3]);
    void (*assignMass) (float, float, float, int, int, M3fType&);
    if (massOption == 0) {
        printf("Nearest Grid Point.\n");
        assignMass = &NGP;
    }
    else if (massOption == 1) {
        printf("Cloud in Cell.\n");
        assignMass = &CIC;
    }
    else if(massOption == 2) {
        printf("Triangle Shaped Cloud.\n");
        assignMass = &TSC;
    }
    else if (massOption == 3) {
        printf("Piecewise Cubic Spline.\n");
        assignMass = &PCS;
    }

    int size, my_rank;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    ptrdiff_t alloc_local, slab_size, slab_start_index;
    fftw_mpi_init();

    //determine slab size and offset in x-axis
    alloc_local = fftw_mpi_local_size_3d(nGrid, nGrid, nGrid, MPI_COMM_WORLD, &slab_size, &slab_start_index);
    assert(slab_size > 0);
    printf("Rank%d: slab_size = %d, slab starts at %d\n", my_rank, slab_size, slab_start_index);
    int* COMM_SLAB_SIZE = new int [size];
    int* COMM_SLAB_START = new int [size];
    int* SLAB2RANK = new int[nGrid];
    MPI_Allgather(&slab_size, 1, MPI_INTEGER, COMM_SLAB_SIZE, 1, MPI_INTEGER, MPI_COMM_WORLD);
    MPI_Allgather(&slab_start_index, 1, MPI_INTEGER, COMM_SLAB_START, 1, MPI_INTEGER, MPI_COMM_WORLD);
    
    int curPos = 0;
    for (int i = 0; i < nGrid; i++) {
	if (curPos < size - 1 && i >= COMM_SLAB_START[curPos+1]) curPos++;    
        SLAB2RANK[i] = curPos;
    }

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    std::uint64_t N = io.count();

    // Load particle positions
    std::cerr << "Total " << N << " particles" << std::endl;
    int nPer = (N + size - 1) / size;
    if (static_cast<unsigned long>(size) >= (nPer + N % size)) nPer--;
    int iStart = my_rank * nPer;
    int iEnd = (my_rank == size - 1) ? N : (my_rank + 1) * nPer;
    assert(iStart < N);
    assert(iEnd <= N);
    // range: [iStart, iEnd)
    printf("rank:%d, range:(%d, %d)\n", my_rank, iStart, iEnd);
    M2fType r(blitz::Range(iStart, iEnd - 1), blitz::Range(0,2));
    timePointType start = std::chrono::system_clock::now();
    io.load(r);
    timePointType end = std::chrono::system_clock::now();
    durationType duration = end - start;
    printf("Reading file took: %.8f s\n", duration);
    
    coord_Type* coord = reinterpret_cast<coord_Type*>(r.data());
    
    /*
      |______________|____________|_____________|___________|_____________|
      ^   rank0      ^    rank1   ^    rank2    ^   rank3   ^   rankN     ^
      0         cutPoints[1] cutPoints[2]  cutPoints[3] cutPoints[N]   iEnd - iStart

      Use partition func to determine the first point with greater grid Index COMM_SLAB_START[i]
     */


    int* cutPoints = new int[size+1];
    cutPoints[0] = 0;
    cutPoints[size] = iEnd - iStart;
    int curStart = 0;
    
    for (int i = 1; i < size; i++) {
           int cutPoint = partition(coord, curStart, iEnd - iStart, COMM_SLAB_START[i], nGrid);
           curStart = cutPoint;
           cutPoints[i] = cutPoint;
    }

    // check if ascending
    for (int i = 1; i < size + 1; i++) {
        assert(cutPoints[i-1] <= cutPoints[i]);
	}
   
    //check points in range has corresponding gridIndex
    curStart = 0;
    for (int i = 1; i < size; i++) {
        int cutPoint = cutPoints[i];
        //printf("slab %d starts at %d\n", COMM_SLAB_START[i], cutPoint);
        for (int s = 0; s < cutPoint; s++) assert(std::floor((coord[s][0] + 0.5) * nGrid) < COMM_SLAB_START[i]);
        for (int s = cutPoint; s < iEnd - iStart; s++) assert(std::floor((coord[s][0] + 0.5) * nGrid) >= COMM_SLAB_START[i]);
    }
    // difference cutPoint -> number of particles to send to each rank
    int* num_Particles_toSend = new int[size];
    int* num_Particles_toRecv = new int[size];
    for (int i = 0; i < size; i++) num_Particles_toSend[i] = cutPoints[i+1] - cutPoints[i];
    // check sum of particles to send == #particles loaded
    int res = 0;
    for (int i = 0; i < size; i++) res += num_Particles_toSend[i];
    assert(res == (iEnd - iStart));
    /*
    printf("rank%d sent:", my_rank);
    for (int i = 0; i < size; i++) {
        printf(" %d,", num_Particles_toSend[i]);
        }
    printf("\n");
    */
    //broadcast num_particles to send to all ranks
    MPI_Alltoall(num_Particles_toSend, 1, MPI_INT, num_Particles_toRecv, 1, MPI_INT, MPI_COMM_WORLD);
    /*
    printf("rank%d received:", my_rank);
    for (int i = 0; i < size; i++) {
	printf(" %d,", num_Particles_toRecv[i]);
	}
    printf("\n");
    */
    // determine total particles received by all ranks
    int new_num_Particle = 0;
    for (int i = 0; i < size; i++) new_num_Particle += num_Particles_toRecv[i];
    // check if all particles have been sent
    int newSumCheck;
    MPI_Allreduce(&new_num_Particle, &newSumCheck, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    assert(newSumCheck == N);
    
    // calculate send Count and offset in quatity of float numbers
    float* sorted_Particles = new float[new_num_Particle * 3];
    int* MPISendCount = new int[size];
    int* MPIRecvCount = new int[size];
    for (int i = 0; i < size; i++) {
	MPISendCount[i] = (num_Particles_toSend[i] * 3);
	MPIRecvCount[i] = (num_Particles_toRecv[i] * 3);
	assert(MPISendCount[i] >= 0);
	assert(MPIRecvCount[i] >= 0);
	}
    
    int* MPISendoffset = new int[size];
    MPISendoffset[0] = 0;
    for (int i = 1; i < size; i++) MPISendoffset[i] = MPISendoffset[i-1] + MPISendCount[i-1];
    int* MPIRecvoffset = new int[size];
    MPIRecvoffset[0] = 0;
    for (int i = 1; i < size; i++) MPIRecvoffset[i] = (MPIRecvoffset[i-1]) + MPIRecvCount[i-1];



    /*
    printf("rank%d sendCount: ", my_rank);
    for (int i = 0; i < size; i++) printf("%d, ", MPISendCount[i]);
    printf("\n rank%d  sendoffset: ", my_rank);
    for (int i = 0; i < size; i++) printf("%d, ", MPISendoffset[i]);
    printf("\n rank%d recvCount: ", my_rank);
    for (int i = 0; i < size; i++) printf("%d, ", MPIRecvCount[i]);
    printf("\n rank%d recvOffset: ", my_rank);
    for (int i = 0; i < size; i++) printf("%d, ", MPIRecvoffset[i]);
    printf("\n");
    */
    // send particles to corresponding ranks
    MPI_Alltoallv(r.data(), MPISendCount, MPISendoffset, MPI_FLOAT, sorted_Particles, MPIRecvCount, MPIRecvoffset, MPI_FLOAT, MPI_COMM_WORLD);   
    delete [] MPISendoffset;
    delete [] MPIRecvoffset;
    delete [] MPIRecvCount;
    delete [] MPISendCount;
    
    delete [] num_Particles_toSend;
    delete [] num_Particles_toRecv;

    // rSorted stores the particles that belongs to the rank in physical space (put float* in)
    M2fType rSorted(sorted_Particles, blitz::shape(new_num_Particle, 3), blitz::neverDeleteData);
    // check if slab index is correct
    for (int i = 0; i < new_num_Particle; i++) {
	    int curSlab = std::floor((rSorted(i,0) + 0.5) * nGrid);
	    if (my_rank < size - 1) assert(curSlab < COMM_SLAB_START[my_rank + 1]);
	    assert(curSlab >= COMM_SLAB_START[my_rank]);
    }
    printf("rank%d has %lu sorted particles ready for mass assignment\n", my_rank, new_num_Particle);   
    

    /*
     massOption = 0:NGP; 1:CIC; 2:TSC; 3:PCS
     padding = 0:NGP; 2:CIC; 2:TSC; 4:PCS
     size of x (firstDim) = slab_size(fftw_mpi_localsize) + xPadding
     padded grid size for massAssignment: firstDim * nGrid * (nGrid + 2)
     padded grid index: [0, firstDim) x [0, nGrid) * [0, nGrid+2)
     */
    int xPadding = 0;
    int xOffset = 0;
    if (massOption == 1 || massOption == 2) {
	xPadding = 2;
	xOffset = 1;
    }
    else if (massOption == 3) {
	xPadding = 4;
	xOffset = 2;
    }
    int firstDim = slab_size + xPadding;
    float* datawPadding = new (std::align_val_t(64)) float[firstDim * nGrid * (nGrid + 2)];
    M3fType gridwPadding(datawPadding, blitz::shape(firstDim, nGrid, nGrid+2), blitz::neverDeleteData);
    gridwPadding = 0.0;
    M3fType grid = gridwPadding(blitz::Range::all(), blitz::Range::all(), blitz::Range(0, nGrid-1));
    M3fType paddingU = grid(blitz::Range(0, xOffset - 1), blitz::Range::all(), blitz::Range::all());
    M3fType paddingD = grid(blitz::Range(xOffset + slab_size, firstDim - 1), blitz::Range::all(), blitz::Range::all());
    
    M3fType gridwMass = grid(blitz::Range(xOffset, xOffset + slab_size - 1), blitz::Range::all(), blitz::Range::all());
    M3fType gridFFT = gridwPadding(blitz::Range(xOffset, xOffset + slab_size - 1), blitz::Range::all(), blitz::Range::all());

    //std::complex<float>* dataComplex = reinterpret_cast<std::complex<float>*>(datawPadding);
   //M3cType kGrid(dataComplex, blitz::shape(firstDim, nGrid, nGrid / 2 + 1));
    start = std::chrono::system_clock::now();

    printf("rank%d pU:%d, %d, pD:%d, %d\n", my_rank, 0, xOffset - 1, xOffset + slab_size, slab_size + xPadding - 1);
    int uBound = (my_rank == size - 1) ? nGrid : COMM_SLAB_START[my_rank+1];
    // upperBoundary = the physical wall in x-axis that cuts the slab
    float upperBoundary = 1.0 / nGrid * uBound;
    printf("rank%d, upperBoundary%f\n", my_rank, upperBoundary);

// mass Assignment  
#pragma omp parallel
{
#pragma omp for
    for (int pn = 0; pn < new_num_Particle; pn++) {
        float x = (rSorted(pn, 0) + 0.5);
        float y = (rSorted(pn, 1) + 0.5);
        float z = (rSorted(pn, 2) + 0.5);
	// if too close to the boudary, adjust
        if (abs(x-upperBoundary) < 0.0001) x -= 0.0001;
        if (abs(y-1.0) < 0.0001) y -= 0.0001;
        if (abs(z-1.0) < 0.0001) z -= 0.0001;
        // adjust x-axis gridIndex of particle to [xOffset, xOffset+slab_size-1]
	x *= nGrid;
	x -= COMM_SLAB_START[my_rank];
        x += xOffset;
	y *= nGrid;
	z *= nGrid;
        assert(x >= grid.lbound(0) + xOffset);
        assert(static_cast<int>(std::floor(x)) <= grid.ubound(0)-xOffset);
        assert(y >= 0 && y < nGrid);
        assert(z >= 0 && z < nGrid);
        assignMass(x, y, z, nGrid, grid.ubound(0), grid);
    }
    
}
    
    end = std::chrono::system_clock::now();
    duration = end - start;
    printf("Mass assignment took: %.8f s\n", duration);
    
//ghost region communication 
    if (massOption > 0) {
	int ghostSendCount = xOffset * nGrid * nGrid;
        int color1 = my_rank / 2;
	int key1 = my_rank & 1;
	int color2 = ((my_rank - 1 + size) % size) / 2;
	int key2 = !(my_rank & 1);
	int color3 = MPI_UNDEFINED;
	int key3;
	if (size % 2) {
	    if (my_rank == 0) {
	        color2 = MPI_UNDEFINED;
		color3 = 0;
		key3 = 0;
	    }
	    if (my_rank == size - 1) { 
		key3 = 1;
	    }
	}
        printf("rank:%d, (c1,k1):(%d,%d), (c2,k2):(%d,%d)\n", my_rank, color1, key1, color2, key2);

        MPI_Comm newcomm1, newcomm2, newcomm3;
        MPI_Comm_split(MPI_COMM_WORLD, color1, key1, &newcomm1);
	MPI_Comm_split(MPI_COMM_WORLD, color2, key2, &newcomm2);
	MPI_Comm_split(MPI_COMM_WORLD, color3, key3, &newcomm3);
	int size1, size2, size3, rank1, rank2, rank3;
	
	if (size % 2 == 0 || (my_rank !=0 && my_rank != size - 1)) {
	    MPI_Comm_size(newcomm1, &size1);
 	    MPI_Comm_size(newcomm2, &size2);
	    MPI_Comm_rank(newcomm1, &rank1);
	    MPI_Comm_rank(newcomm2, &rank2);
	    assert(size1 == 2 && size2 == 2);
	    printf("rank%d: %d, %d\n",my_rank, rank1, rank2);
	}
	
	else {
	    MPI_Comm_size(newcomm3, &size3);
	    MPI_Comm_rank(newcomm3, &rank3);
	    assert(size3 == 2);
	    if (my_rank == 0) {
	        MPI_Comm_size(newcomm1, &size1);
		MPI_Comm_rank(newcomm1, &rank1);
		assert(size1 == 2);
		//printf("rank0: %d, %d\n", rank1, rank3);
	    }
	    else {
		MPI_Comm_size(newcomm2, &size2);
		MPI_Comm_rank(newcomm2, &rank2);
		assert(size2 == 2);
		//printf("rank%d: %d, %d\n", size - 1, rank2, rank3);
	    }
	}

	float* ghostDS = paddingD.data(); 
	float* ghostUS = paddingU.data();
	float* ghostDR = new float [ghostSendCount];
	float* ghostUR = new float [ghostSendCount];
	int tag1 = 42;
	int neighbor1 = (rank1 == 0) ? 1 : 0;
        float* ghostSend1 = (rank1 == 0) ? ghostDS : ghostUS;
	float* ghostRecv1 = (rank1 == 0) ? ghostDR : ghostUR;

	MPI_Request* request1 = new MPI_Request[2];
	MPI_Isend(ghostSend1, ghostSendCount, MPI_FLOAT, neighbor1, tag1, newcomm1, &request1[0]);
	MPI_Irecv(ghostRecv1, ghostSendCount, MPI_FLOAT, neighbor1, tag1, newcomm1, &request1[1]);
	MPI_Waitall(2, request1, MPI_STATUS_IGNORE);
       //printf("com1: rank%d,send%f, recv%f\n",my_rank, ghostSend1[0], ghostRecv1[0]);
	
	int tag2 = 58;
	int neighbor2 = (rank2 == 0) ? 1 : 0;
        float* ghostSend2 = (rank2 == 0) ? ghostDS: ghostUS;
	float* ghostRecv2 = (rank2 == 0) ? ghostDR : ghostUR;

	MPI_Request* request2 = new MPI_Request[2];
	MPI_Isend(ghostSend2, ghostSendCount, MPI_FLOAT, neighbor2, tag2, newcomm2, &request2[0]);
	MPI_Irecv(ghostRecv2, ghostSendCount, MPI_FLOAT, neighbor2, tag2, newcomm2, &request2[1]);
	MPI_Waitall(2, request2, MPI_STATUS_IGNORE);
	//printf("com2: rank%d, send%f, recv%f\n", my_rank, ghostSend2[0], ghostRecv2[0]);

        M3fType uRegion2Add = grid(blitz::Range(xOffset, xOffset * 2 - 1), blitz::Range::all(), blitz::Range::all());
	M3fType dRegion2Add = grid(blitz::Range(slab_size, xOffset + slab_size - 1), blitz::Range::all(), blitz::Range::all());
        M3fType uRegionAdd(ghostUR, blitz::shape(xOffset, nGrid, nGrid), blitz::neverDeleteData);
	M3fType dRegionAdd(ghostDR, blitz::shape(xOffset, nGrid, nGrid), blitz::neverDeleteData);
	uRegion2Add += uRegionAdd;
	dRegion2Add += dRegionAdd;
        
	/*
	float ghostDSSum = blitz::sum(paddingD);
	float ghostUSSum = blitz::sum(paddingU);
        float ghostDRSum = blitz::sum(dRegionAdd);
	float ghostURSum = blitz::sum(uRegionAdd);
	printf("rank%d DSend:%f, USend:%f, DRecv:%f, URecv:%f\n", my_rank, ghostDSSum, ghostUSSum, ghostDRSum, ghostURSum);
	*/
	
	delete [] request2;
        delete [] request1;
        delete [] ghostDR;
	delete [] ghostUR;
        
    }
    
    

    // calculate over density
    float blitzSum = blitz::sum(gridwMass);
    float average_density = blitzSum / (slab_size * nGrid * nGrid);
    printf("sum = %f, actualSize = %d\n",blitzSum, slab_size * nGrid * nGrid);
    printf("average_density = %f\n", average_density);
    
    float massAssignmentCheck; 
    MPI_Allreduce(&blitzSum, &massAssignmentCheck, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    printf("massAssignmentCheck:%.8f\n", massAssignmentCheck);

    gridwMass -= average_density;
    gridwMass /= average_density;
    printf("overall_density = %f\n", blitz::sum(gridwMass));

 

        start = std::chrono::system_clock::now();
	//does not work with inplace fftw (double free or corruption (!prev))
   	//fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(nGrid, nGrid, nGrid, datawPadding, (fftwf_complex *)dataComplex,MPI_COMM_WORLD, FFTW_ESTIMATE);
	// works with complex data allocated by fftw_alloc_complex but much slower than non-mpi fftw calls
	fftwf_complex* fftw_complex_data;
	fftw_complex_data = fftwf_alloc_complex(alloc_local);
	fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(nGrid, nGrid, nGrid, gridFFT.data(), fftw_complex_data,MPI_COMM_WORLD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    	end = std::chrono::system_clock::now();
    	duration = end - start;
    	printf("FFT took: %.8f s\n", duration);	
    
    delete [] cutPoints;
    delete [] COMM_SLAB_SIZE;
    delete [] COMM_SLAB_START;
    delete [] SLAB2RANK;
    fftw_mpi_cleanup();
    delete [] datawPadding;
    delete [] sorted_Particles;
    MPI_Finalize();
    return 0;
}

void func() {
#ifdef _OPENMP
    bool p = omp_in_parallel();
    printf("%s\n", p ? "parallel" : "serial");
#endif
}

void NGP(float x, float y, float z, int nGrid, int xGrid, M3fType& grid) {
#pragma omp parallel
{
    int i = std::floor(x);
    int j = std::floor(y);
    int k = std::floor(z);
#pragma omp atomic
    grid(i, j, k) += 1.0;
}
}

void CIC(float x, float y, float z, int nGrid, int xGrid, M3fType& grid) {
#pragma omp parallel
{
    float Wx[2];
    float Wy[2];
    float Wz[2];
    int ix = cic_weights(x, Wx);
    int iy = cic_weights(y, Wy);
    int iz = cic_weights(z, Wz);
    float totalW = 0.0;
#pragma omp for collapse(3)
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int indexI = ix + i;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI <= xGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
               grid(indexI, indexJ, indexK) += W;
               totalW += W;
            }
        }
    }
   assert(abs(totalW - 1.0) < 0.0001);
}
    
}

void TSC(float x, float y, float z, int nGrid, int xGrid, M3fType& grid) {
#pragma omp parallel
{
    float Wx[3];
    float Wy[3];
    float Wz[3];
    int ix = tsc_weights(x, Wx);
    int iy = tsc_weights(y, Wy);
    int iz = tsc_weights(z, Wz);
    float totalW = 0.0;
#pragma omp for collapse(3)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int indexI = ix + i;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI <= xGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
                grid(indexI, indexJ, indexK) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.0001);
}
    
}

void PCS(float x, float y, float z, int nGrid, int xGrid, M3fType& grid) {
#pragma omp parallel
{
    float Wx[4];
    float Wy[4];
    float Wz[4];
    int ix = pcs_weights(x, Wx);
    int iy = pcs_weights(y, Wy);
    int iz = pcs_weights(z, Wz);
    float totalW = 0.0;
#pragma omp for collapse(3)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                int indexI = ix + i;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI <= xGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
                grid(indexI, indexJ, indexK) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.00001);
}
    
}

float PCS_return(float x, float y, float z, int nGrid, int xGrid, M3fType& grid) {
    float Wx[4];
    float Wy[4];
    float Wz[4];
    int ix = pcs_weights(x, Wx);
    int iy = pcs_weights(y, Wy);
    int iz = pcs_weights(z, Wz);
    float totalW = 0.0;
#pragma omp parallel for collapse(3)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                int indexI = (ix + i + nGrid) % nGrid;
                int indexJ = (iy + j + nGrid) % nGrid;
                int indexK = (iz + k + nGrid) % nGrid;
                assert(indexI >= 0 && indexI < nGrid);
                assert(indexJ >= 0 && indexJ < nGrid);
                assert(indexK >= 0 && indexK < nGrid);
                float W = Wx[i] * Wy[j] * Wz[k];
#pragma omp atomic
                grid(indexI, indexJ, indexK) += W;
                totalW += W;
            }
        }
    }
    assert(abs(totalW - 1.0) < 0.0001);
    return totalW;
}


void outPut(const char* fileName, int nGrid, const M2fType& projected) {
    std::ofstream fout(fileName);
    if (fout.is_open()) {
        for (int i = 0; i < nGrid; i++) {
            for (int j = 0; j < nGrid; j++) {
                fout << projected(i, j) << ",";
            }
            fout << std::endl;
        }
    }
    fout.close();
}

void LinearBinning(int nGrid, const M3cType& kGrid) {
    blitz::Array<unsigned int, 1> nPower(nGrid);
    blitz::Array<double, 1> fPower(nGrid);
    nPower = 0;
    fPower = 0.0;

    for (int i = 0; i < nGrid; i++) {
            int kx = (i <= nGrid/2) ? i : -nGrid + i;
            for (int j = 0; j < nGrid; j++) {
                    int ky = (j <= nGrid/2) ? j : -nGrid + j;
                    for (int k = 0; k < (nGrid / 2 + 1); k++) {
                            int kz = k;
                            double kf = std::sqrt(kx * kx + ky * ky + kz * kz);
                            int kIndex = std::floor(kf);
                            assert(kIndex < nGrid);
                            nPower(kIndex)++;
                            fPower(kIndex) += std::norm(kGrid(i, j, k));
                        }
            }

    }
    for (int i = 0; i < nGrid; i++) {
            if (nPower(i)) fPower(i) /= nPower(i);
    }

    std::string fileName = "Linear_";
    fileName += std::to_string(nGrid);
    fileName += "mpi.txt";
    std::ofstream fout(fileName.c_str());
    if (fout.is_open()) {
            for (int i = 0; i < nGrid; i++) {
                    fout << fPower(i) << std::endl;
                }
    }

}

void VariableBinning(int nGrid, int nBins, const M3cType& kGrid) {
    blitz::Array<unsigned long, 1> nPower(nBins);
    blitz::Array<double, 1> fPower(nBins);
    nPower = 0;
    fPower = 0.0;
    int maxBin = std::floor(std::sqrt(3) * nGrid / 2.0);
    printf("Variable maxBin = %d\n", maxBin);

    for (int i = 0; i < nGrid; i++) {
            int kx = (i <= nGrid/2) ? i : -nGrid + i;
            for (int j = 0; j < nGrid; j++) {
                    int ky = (j <= nGrid/2) ? j : -nGrid + j;
                    for (int k = 0; k < (nGrid / 2 + 1); k++) {
                            int kz = k;
                            double kf = std::sqrt(kx * kx + ky * ky + kz * kz);
                            int kIndex = std::floor(kf / maxBin * nBins);
                            if (kf / maxBin >= 1) kIndex--;
			   
			    assert(kIndex < nBins);
                            nPower(kIndex)++;
                            fPower(kIndex) += std::norm(kGrid(i, j, k));
			    //printf("kf:%f, kIndex:%d, power:%f\n",kf, kIndex, std::norm(kGrid(i, j, k)));
                        }
            }

    }
    for (int i = 0; i < nBins; i++) {
            if (nPower(i)) fPower(i) /= nPower(i);
    }

    std::string fileName = "Variable";
    fileName += "_";
    fileName += std::to_string(nGrid);
    fileName += "_";
    fileName += std::to_string(nBins);
    fileName += "mpi.txt";
    std::ofstream fout(fileName.c_str());
    if (fout.is_open()) {
            for (int i = 0; i < nBins; i++) {
                    fout << fPower(i) << std::endl;
                }
    }
}

void LogBinning(int nGrid, int nBins, const M3cType& kGrid) {
    blitz::Array<unsigned long, 1> nPower(nBins);
    blitz::Array<double, 1> fPower(nBins);
    nPower = 0;
    fPower = 0.0;
    int maxBin = std::floor(std::sqrt(3) * nGrid / 2.0);
    printf("Log maxBin = %d\n", maxBin);

    for (int i = 0; i < nGrid; i++) {
            int kx = (i <= nGrid/2) ? i : -nGrid + i;
            for (int j = 0; j < nGrid; j++) {
                    int ky = (j <= nGrid/2) ? j : -nGrid + j;
                    for (int k = 0; k < (nGrid / 2 + 1); k++) {
                            int kz = k;
			    if (kx == 0 && ky == 0 && kz == 0) continue;
                            double kf = std::sqrt(kx * kx + ky * ky + kz * kz);
                            int kIndex = std::floor(std::log(kf) / std::log(maxBin) * nBins);
                            if (kf / maxBin >= 1) kIndex--;

                            assert(kIndex < nBins);
                            nPower(kIndex)++;
                            fPower(kIndex) += std::norm(kGrid(i, j, k));
                           //printf("kf:%f, kIndex:%d, power:%f\n",kf, kIndex, std::norm(kGrid(i, j, k)));
                        }
            }

    }

    for (int i = 0; i < nBins; i++) {
            if (nPower(i)) fPower(i) /= nPower(i);
    }

    std::string fileName = "Log";
    fileName += "_";
    fileName += std::to_string(nGrid);
    fileName += "_";
    fileName += std::to_string(nBins);
    fileName += "mpi.txt";
    std::ofstream fout(fileName.c_str());
    if (fout.is_open()) {
            for (int i = 0; i < nBins; i++) {
                    fout << fPower(i) << std::endl;
                }
    }

}


void swapCoord(coord_Type* data, int i, int j) {
    coord_Type tmp;
    tmp[0] = data[i][0];
    tmp[1] = data[i][1];
    tmp[2] = data[i][2];

    data[i][0] = data[j][0];
    data[i][1] = data[j][1];
    data[i][2] = data[j][2];

    data[j][0] = tmp[0];
    data[j][1] = tmp[1];
    data[j][2] = tmp[2];
}

int partition(coord_Type* data, int leftIndex, int rightIndex, int slabIndex, int nGrid) {
    auto coord2Slab = [] (float x, int nGrid) -> int {return std::floor((x + 0.5) * nGrid);};
    int i = leftIndex;
    int j = rightIndex - 1;

    while (i <= j) {
        if (coord2Slab(data[i][0], nGrid) < slabIndex) ++i;
      else break;
    }
    while (i <= j) {
      if (coord2Slab(data[j][0], nGrid) >= slabIndex) --j;
      else break;
    }
    if (i < j) {
        swapCoord(data, i, j);
        while (1) {
          ++i;
          while (coord2Slab(data[i][0], nGrid) < slabIndex) ++i;
          --j;
          while (coord2Slab(data[j][0], nGrid) >= slabIndex) --j;
          if (i < j) {
              swapCoord(data, i, j);
        }
        else break;
      }
    }
    return i;
}


