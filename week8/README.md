mpirun -n 2 assign B100 .00100 will not divide up the work automatically. Instead, each process will have the full-size data, and replicate the work.

After utilizing MPI, it takes significantly more time for each process to read the data and the mass assignment becomes slower also.
