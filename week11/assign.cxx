// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include <chrono>
#include <blitz/array.h>
#include "tipsy.h"
#include <math.h>
#include <new>
#include "weights.h"
#include <vector>
#include <complex>
#include <fftw3.h>
#include <numeric>
#include <mpi.h>
#include <fftw3-mpi.h>
#include "gpufft.h"
#include <algorithm>
#include <cuda.h>
#include <cufft.h>

int wrap_edge(int coordinate, int N)
{
    if (coordinate < 0)
    {
        coordinate += N;
    }
    else if (coordinate >= N)
    {
        coordinate -= N;
    }
    return coordinate;
}

int add_order(int coordinate, int order)
{
    return coordinate + order;
}

int precalculate_W(float W[], int order, float r, float cell_half = 0.5)
{
    switch (order)
    {
    case 1:
        return ngp_weights(r, W);
    case 2:
        return cic_weights(r, W);
    case 3:
        return tsc_weights(r, W);
    case 4:
        return pcs_weights(r, W);
    default:
        throw std::invalid_argument("[precalculate_W] order out of bound");
    }
}

int k_indx(int i, int nGrid)
{
    int res = i;
    if (i > nGrid / 2)
    {
        res = i - nGrid;
    }
    return res;
}

int get_i_bin(double k, int n_bins, int nGrid, int task = 1)
{
    double k_max = sqrt((nGrid / 2.0) * (nGrid / 2.0) * 3.0);
    switch (task)
    {
    case 1:
        return int(k);
    case 2:
        return int(k / k_max * n_bins);
    case 3:
        if (k != 0)
        {
            return int(log10(k) / log10(k_max) * n_bins);
        }
        else
        {
            return 0;
        }
    default:
        std::cout << "[get_i_bin] Selected invalid task number. Deafult to 1" << std::endl;
        return int(k);
    }
}

void save_binning(const int binning, std::vector<double> &fPower, std::vector<int> &nPower)
{
    const char *binning_filename;
    switch (binning)
    {
    case 1:
        binning_filename = "/users/okondrat/ex10/linear_binning.csv";
        break;
    case 2:
        binning_filename = "variable_binning.csv";
        break;
    case 3:
        binning_filename = "log_binning.csv";
        break;
    }

    std::ofstream f1(binning_filename);
    f1.precision(6);
    f1 << "P_k,k,n" << std::endl;
    for (int i = 0; i < fPower.size(); ++i)
    {
        f1 << (nPower[i] != 0 ? (fPower[i] / nPower[i]) : 0) << "," << i + 1 << "," << nPower[i] << std::endl;
    }
    f1.close();
}

void assign_mass(blitz::Array<float, 2> &r, int part_i_start, int part_i_end, int nGrid, blitz::Array<float, 3> &grid, int order, int grid_start, int grid_end)
{
    // Loop over all cells for this assignment
    float cell_half = 0.5;
    printf("Assigning mass to the grid using order %d\n", order);
#pragma omp parallel for
    for (int pn = part_i_start; pn < part_i_end; ++pn)
    {
        float x = r(pn, 0);
        float y = r(pn, 1);
        float z = r(pn, 2);

        float rx = (x + 0.5) * nGrid;
        float ry = (y + 0.5) * nGrid;
        float rz = (z + 0.5) * nGrid;

        // precalculate Wx, Wy, Wz and return start index
        float Wx[order], Wy[order], Wz[order];
        int i_start = precalculate_W(Wx, order, rx);
        int j_start = precalculate_W(Wy, order, ry);
        int k_start = precalculate_W(Wz, order, rz);
        if (i_start < 0)
        {
            i_start += nGrid;
        }

        for (int i = i_start; i < i_start + order; i++)
        {
            for (int j = j_start; j < j_start + order; j++)
            {
                for (int k = k_start; k < k_start + order; k++)
                {
                    float W_res = Wx[i - i_start] * Wy[j - j_start] * Wz[k - k_start];
                    if (i < grid_start || i >= grid_end)
                    {
                        printf("[ERROR] i = %d, grid_start = %d, i_start = %d \n", i, grid_start, i_start);
                    }
// Deposit the mass onto grid(i,j,k)
#pragma omp atomic
                    grid(i, wrap_edge(j, nGrid), wrap_edge(k, nGrid)) += W_res;
                }
            }
        }
    }
}

void project_grid(blitz::Array<float, 3> &grid, int nGrid, const char *out_filename)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    blitz::Array<float, 2> projected(nGrid, nGrid);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            projected(i, j) = max(grid(i, j, blitz::Range::all()));
        }
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Projection took " << std::setw(9) << diff.count() << " s\n";

    std::ofstream f(out_filename);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            if (j != nGrid - 1)
            {
                f << projected(i, j) << ",";
            }
            else
            {
                f << projected(i, j);
            }
        }
        f << std::endl;
    }
    f.close();
}

struct particle
{
    float x, y, z;
};

bool compare_particles(const particle &a, const particle &b)
{
    return (a.x < b.x);
};

struct stream_info {
    void* slab_gpu;
    void* work_gpu;
    cudaStream_t* stream;
};



int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int i_rank, N_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N_rank);

    if (argc <= 1)
    {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std grid-size [order, projection-filename]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc > 2)
    {
        nGrid = atoi(argv[2]);
    }

    int order = 4;
    if (argc > 3)
    {
        order = atoi(argv[3]);
    }

    const char *out_filename = (argc > 4) ? argv[4] : "projected.csv";

    // Init FFTW
    fftwf_mpi_init();
    ptrdiff_t start0, local0;
    ptrdiff_t start1, local1;
    ptrdiff_t n_transpose[] = {nGrid, nGrid};
    auto alloc_local = fftwf_mpi_local_size_many_transposed(2, n_transpose, nGrid + 2,
                                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD,
                                                            &local0, &start0, &local1, &start1);
    // Collect all start0 and local0
    std::vector<int> all_start0(N_rank);
    std::vector<int> all_local0(N_rank);
    MPI_Allgather(&start0, 1, MPI_INT, &all_start0[0], 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local0, 1, MPI_INT, &all_local0[0], 1, MPI_INT, MPI_COMM_WORLD);

    // Fill particles x position to rank
    blitz::Array<int, 1> slab2rank(nGrid);
    for (int i = 0; i < N_rank; ++i)
    {
        for (int j = all_start0[i]; j < all_start0[i] + all_local0[i]; ++j)
        {
            slab2rank(j) = i;
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail())
    {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    int N = io.count();

    // Load particle positions
    int N_per = (N + N_rank - 1) / N_rank;
    int i_start = N_per * i_rank;
    int i_end = std::min(N_per * (i_rank + 1), N);

    std::cerr << "Loading " << N_per << " particles" << std::endl;
    blitz::Array<float, 2> r(blitz::Range(i_start, i_end - 1), blitz::Range(0, 2));
    io.load(r);
    std::chrono::duration<double> diff_load = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Reading file took " << std::setw(9) << diff_load.count() << " s\n";

    particle *prows = reinterpret_cast<particle *>(r.data());
    std::sort(prows, prows + r.rows(), [order, nGrid](const particle &a, const particle &b)
              { 
                float a_start = ((a.x + 0.5) * nGrid - (order - 1) * 0.5);
                float b_start = ((b.x + 0.5) * nGrid - (order - 1) * 0.5);
                a_start = a_start < 0 ? a_start + nGrid : a_start;
                b_start = b_start < 0 ? b_start + nGrid : b_start;
                return (a_start < b_start); });

    std::vector<int> send_counts(N_rank, 0);
    int rank_to_send = 0;
    for (int pn = i_start; pn < i_end; ++pn)
    {
        float x = r(pn, 0);
        int i_start = floorf((x + 0.5) * nGrid - (order - 1) * 0.5);
        int rank_to_send = 0;
        if (i_start < 0)
        {
            rank_to_send = N_rank - 1;
        }
        else
        {
            rank_to_send = slab2rank(i_start);
        }
        send_counts[rank_to_send]++;
    }

    for (int i = 0; i < N_rank; i++)
    {
        send_counts[i] *= 3;
    }

    std::vector<int> send_displs(N_rank);
    send_displs[0] = 0;
    for (int i = 1; i < N_rank; ++i)
    {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    }

    // Compute recv counts and displacements
    std::vector<int> recv_counts(N_rank);
    std::vector<int> recv_displs(N_rank);
    MPI_Alltoall(&send_counts[0], 1, MPI_INT, &recv_counts[0], 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    for (int i = 0; i < N_rank; ++i)
    {
        if (i == 0)
            recv_displs[i] = 0;
        else
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        total_recv += recv_counts[i];
    }
    int part_count = total_recv / 3;

    printf("[Rank %d] Total recv = %d\n", i_rank, part_count);
    blitz::Array<float, 2> r_local(part_count, 3);
    MPI_Alltoallv(r.dataFirst(), &send_counts[0], &send_displs[0], MPI_FLOAT,
                  r_local.data(), &recv_counts[0], &recv_displs[0], MPI_FLOAT,
                  MPI_COMM_WORLD);

    printf("[Rank %d] first particle = %f, last particle = %f\n", i_rank, r_local(0, 0), r_local(part_count - 1, 0));
    int grid_start = start0;
    int grid_end = std::min(int(start0 + local0), nGrid) + order - 1;

    // Complex grid creation with local1 and start1
    int grid_start_complex = start1;
    int grid_end_complex = std::min(int(start1 + local1), nGrid);
    // Make sure that we have room for the ghost region
    alloc_local = std::max(alloc_local, nGrid * (nGrid / 2 + 1) * (local0 + order - 1));
    auto alloc_data = fftwf_alloc_complex(alloc_local);

    // Allocate a slab on the GPU

    float *data = reinterpret_cast<float *>(alloc_data);
    blitz::GeneralArrayStorage<3> storage;
    storage.base() = grid_start, 0, 0;
    blitz::Array<float, 3> grid_data(data, blitz::shape(grid_end - grid_start, nGrid, nGrid + 2), blitz::deleteDataWhenDone, storage);
    grid_data = 0.0;
    blitz::Array<float, 3> grid = grid_data(blitz::Range(grid_start, grid_end - 1), blitz::Range::all(), blitz::Range(0, nGrid - 1));
    blitz::Array<float, 3> ghost_region = grid(blitz::Range(grid_end - order + 1, grid_end - 1), blitz::Range::all(), blitz::Range::all());

    std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);

    blitz::GeneralArrayStorage<3> transposed_storage;
    transposed_storage.ordering() = 2, 0, 1;
    transposed_storage.ascendingFlag() = true;
    transposed_storage.base() = 0, grid_start_complex, 0;

    start_time = std::chrono::high_resolution_clock::now();
    assign_mass(r_local, 0, part_count, nGrid, grid, order, grid_start, grid_end);
    printf("[Rank %d] Grid sum after mass assignment = %f\n", i_rank, sum(grid));
    std::chrono::duration<double> diff_assignment = std::chrono::high_resolution_clock::now() - start_time;
    printf("[Rank %d] Mass assignment took %fs\n", i_rank, diff_assignment.count());

    MPI_Barrier(MPI_COMM_WORLD);

    float actual_mass = sum(grid);
    MPI_Allreduce(MPI_IN_PLACE, &actual_mass, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (i_rank == 0)
    {
        printf("Total mass: %f\n", actual_mass);
    }

    // Exchange the ghost regions
    MPI_Comm comm1, comm2;

    if (order != 1)
    {
        if (N_rank == 1)
        {
            grid(blitz::Range(grid_start, grid_start + order - 2), blitz::Range::all(), blitz::Range::all()) += ghost_region;
        }
        else
        {
            MPI_Request requests[3];
            MPI_Status statuses[3];
            requests[0] = MPI_REQUEST_NULL;
            requests[1] = MPI_REQUEST_NULL;
            requests[2] = MPI_REQUEST_NULL;
            MPI_Comm newcomm1, newcomm2;
            int color = i_rank / 2;
            int key = (i_rank + 1) % 2;

            // Last rank in odd amount of ranks will be alone in communicator so just ignore it
            if (!((N_rank % 2 == 1) && (i_rank == N_rank - 1)))
            {
                MPI_Comm_split(MPI_COMM_WORLD, color, key, &newcomm1);
            }
            else
            {
                MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, i_rank, &newcomm1); // Create an "empty" communicator for ranks not included in the reduction
            }

            if (!((N_rank % 2 == 1) && (i_rank == N_rank - 1)))
            {
                if (key == 0)
                {
                    MPI_Ireduce(MPI_IN_PLACE, grid.data(), (nGrid + 2) * nGrid * (order - 1), MPI_FLOAT, MPI_SUM, 0, newcomm1, &requests[0]);
                }
                else
                {
                    MPI_Ireduce(ghost_region.data(), nullptr, (nGrid + 2) * nGrid * (order - 1), MPI_FLOAT, MPI_SUM, 0, newcomm1, &requests[0]);
                }
            }
            // Second split the communicator into (1,2), (3,4), (5,0)
            if (i_rank == N_rank - 1)
            {
                color = 0;
                key = 1;
            }
            else
            {
                color = (i_rank + 1) / 2;
                key = i_rank % 2;
            }
            // Next to last rank in odd amount of ranks will be alone in communicator so just ignore it
            if (!((N_rank % 2 == 1) && (i_rank == N_rank - 2)))
            {
                MPI_Comm_split(MPI_COMM_WORLD, color, key, &newcomm2);
            }
            else
            {
                MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, i_rank, &newcomm2);
            }
            if (!((N_rank % 2 == 1) && (i_rank == N_rank - 2)))
            {
                if (key == 0)
                {
                    MPI_Ireduce(MPI_IN_PLACE, grid.data(), (nGrid + 2) * nGrid * (order - 1), MPI_FLOAT, MPI_SUM, 0, newcomm2, &requests[1]);
                }
                else
                {
                    MPI_Ireduce(ghost_region.data(), nullptr, (nGrid + 2) * nGrid * (order - 1), MPI_FLOAT, MPI_SUM, 0, newcomm2, &requests[1]);
                }
            }

            if (N_rank % 2 == 1)
            {
                MPI_Comm newcomm3;
                if (i_rank >= N_rank - 2)
                {
                    color = 0;

                    // Odd number of proc
                    if (i_rank == N_rank - 2)
                    {
                        key = 1;
                    }
                    else if (i_rank == N_rank - 1)
                    {
                        key = 0;
                    }
                    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newcomm3);
                }
                else
                {

                    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, i_rank, &newcomm3);
                }
                if (i_rank >= N_rank - 2)
                {
                    if (key == 0)
                    {
                        MPI_Ireduce(MPI_IN_PLACE, grid.data(), (nGrid + 2) * nGrid * (order - 1), MPI_FLOAT, MPI_SUM, 0, newcomm3, &requests[2]);
                    }
                    else
                    {
                        MPI_Ireduce(ghost_region.data(), nullptr, (nGrid + 2) * nGrid * (order - 1), MPI_FLOAT, MPI_SUM, 0, newcomm3, &requests[2]);
                    }
                }
            }
            MPI_Waitall(3, requests, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    actual_mass = sum(grid(blitz::Range(grid_start, grid_end - order), blitz::Range::all(), blitz::Range::all()));
    MPI_Allreduce(MPI_IN_PLACE, &actual_mass, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (i_rank == 0)
    {
        printf("After all reduction grid sum = %f\n", actual_mass);
    }
    // Overdensity
    // grid = grid - 1;

    float diRhoBar = ((1.0f * nGrid * nGrid * nGrid) / actual_mass);
    grid = grid * diRhoBar - 1.0f;
    grid /= (nGrid * nGrid * nGrid);

    start_time = std::chrono::high_resolution_clock::now();
    // Do the 2D transforms
    printf("[Rank %d] Start 1D\n", i_rank);
    

    int n[] = {grid.rows(), grid.cols()}; // 2D FFT of length NxN
    int inembed[] = {grid.rows(), 2 * (grid.cols() / 2 + 1)};
    int onembed[] = {grid.rows(), (grid.cols() / 2 + 1)};
    int howmany = 1;
    int odist = grid.rows() * (grid.cols() / 2 + 1); // Output distance is in "complex"
    int idist = 2 * odist;                           // Input distance is in "real"
    int istride = 1;                                 // Elements of each FFT are adjacent
    int ostride = 1;
    cufftHandle plan;

    size_t workSize;
    cufftCreate(&plan);
    cufftSetAutoAllocation(plan,0);
    cufftMakePlanMany(plan,sizeof(n) / sizeof(n[0]), n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_R2C, howmany, &workSize);
    
    stream_info streams[3];
    for (int i = 0; i < 3; i++) {
	cudaStreamCreate(streams[i].stream);
        streams[i].slab_gpu = allocate_cuda_slab_Stream(nGrid, streams[i].stream);
	streams[i].work_gpu = allocate_cuda_size_Stream(workSize, streams[i].stream);
    }
    
    for (int i = grid_start; i <= grid_end - order; i++)
    {
        blitz::Array<float, 2> slab = grid(i, blitz::Range::all(), blitz::Range::all());
        int numStream = i % 3;
	compute_fft_2D_R2C_Stream(slab, streams[numStream].slab_gpu, &plan, streams[numStream].work_gpu, streams[numStream].stream);
    }
    for (int i = 0; i < 3; i++) {
        destroy_cuda_data_Stream(streams[i].slab_gpu, streams[i].stream);
	destroy_cuda_data_Stream(streams[i].work_gpu, streams[i].stream);
    }
    cufftDestroy(plan);
    printf("[Rank %d] Finish 1D\n", i_rank);
    diff_load = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "2D transform took " << std::setw(9) << diff_load.count() << " s\n";

    // Transpose
    fftwf_plan plan_transpose = fftwf_mpi_plan_many_transpose(
        nGrid, nGrid, nGrid + 2,
        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
        data,
        data,
        MPI_COMM_WORLD, FFTW_ESTIMATE);

    fftwf_execute_r2r(plan_transpose, data, data);

    blitz::Array<std::complex<float>, 3> kdata(complex_data,
                                               blitz::shape(nGrid, grid_end_complex - grid_start_complex, nGrid / 2 + 1),
                                               blitz::neverDeleteData,
                                               transposed_storage);
    kdata.reindexSelf(blitz::shape(0, grid_start_complex, 0));
    // Do the 1D transforms
    constexpr int rank_1d = 1;
    int n_1d[rank_1d] = {nGrid};
    int stride_1d = nGrid / 2 + 1;
    int dist_1d = 1;
    int howmany_1d = nGrid / 2 + 1;
    fftwf_plan plan_1d = fftwf_plan_many_dft(1, n_1d, howmany_1d,
                                             (fftwf_complex *)complex_data, n_1d, stride_1d, dist_1d,
                                             (fftwf_complex *)complex_data, n_1d, stride_1d, dist_1d,
                                             FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i = grid_start_complex; i < grid_end_complex; ++i)
    {
        fftwf_execute_dft(plan_1d,
                          (fftwf_complex *)(&kdata(0, i, 0)),
                          (fftwf_complex *)(&kdata(0, i, 0)));
    }
    printf("[Rank %d] FFT finished \n", i_rank);
    // Linear binning is 1
    // Variable binning is 2
    // Log binning is 3
    const int binning = 1;

    int n_bins = 80;
    if (binning == 1)
    {
        n_bins = nGrid;
    }

    n_bins = 50;
    std::vector<double> fPower(n_bins, 0.0);
    std::vector<int> nPower(n_bins, 0);
    float k_max = sqrt((nGrid / 2.0) * (nGrid / 2.0) * 3.0);

    // loop over δ(k) and compute k from kx, ky and kz
    for (auto ii = kdata.begin(); ii != kdata.end(); ++ii)
    {
        auto bin = [n_bins, nGrid](int k)
        { return k <= n_bins ? k : k - nGrid; };
        auto kx = bin(ii.position()[0]);
        auto ky = bin(ii.position()[1]);
        auto kz = ii.position()[2];
        float k = sqrt(kx * kx + ky * ky + kz * kz);
        auto i = int(k);
        if (i > 0 && i < n_bins)
        {
            fPower[i] += std::norm(*ii);
            nPower[i] += 1;
        }
    }

    if (i_rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, fPower.data(), fPower.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE, nPower.data(), nPower.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(fPower.data(), nullptr, fPower.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nPower.data(), nullptr, nPower.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (i_rank == 0)
    {
        save_binning(binning, fPower, nPower);
    }

    fftwf_destroy_plan(plan_transpose);
    fftwf_destroy_plan(plan_1d);
    MPI_Finalize();
}
