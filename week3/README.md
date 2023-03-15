ALL CODES WERE RUN ON EIGER TO EXPLOIT FULL COMPUTING POWER. TO DO SO, THE BLITZ PART OF THE TEMPLATE CODE IS STRIPPED. THE PERFORMANCE IS COMPARABLE WITH SIMPLE ARRAYS (TESTED ON MY LAPTOP).  

The log-log plot makes more sense when measuring the runtime of different phases. All three phases scale linearly with the number of particles, so x and y axis should be in the same scale. But as the number of particles grows exponentially, it is better to plot in log-log.

In terms of parallelization, OMP boosts the mass assignment process when the number of threads is relatively small. However, the acceleration is not so siginificant when the number of threads exceeds 16. This problem is more severe with larger datasets. OMP is more effective with higher order algorithms. 

