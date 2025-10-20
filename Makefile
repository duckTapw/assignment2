CC = mpicc
CFLAGS = -fopenmp -lm -O3

make: convolution.c 
	$(CC) -o convolution convolution.c $(CFLAGS)
	$(CC) -o convolution_omp convolution_omp.c $(CFLAGS)
	$(CC) -o convolution_mpi convolution_mpi.c $(CFLAGS)
	$(CC) -o convolution_mpi+omp convolution_mpi+omp.c $(CFLAGS)
clean:
	rm convolution convolution_omp convolution_mpi convolution_mpi+omp