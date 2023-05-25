nvcc -O3 -o fftcuda fft10.cu \
-I /store/uzh/uzh8/packages/include \
-I$FFTW_INC -L$FFTW_DIR -lfftw3f -lcufft
