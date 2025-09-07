# Workshop0-IntroPACE

## Installing MPI

First make sure MPI and C are installed on your computer by running
```
mpirun --version
```
If not installed run
```
sudo apt install mpich // linux
brew install open-mpi // mac
```

## Compiling and Running Locally

Creates executable (matmul) that you can run however you like
```
make 
mpirun -n <num processes> ./matmul <matrix_size>
```
To compile and run in one go

NP defaults to 1, and MATRIX_SIZE defaults to 4 times NP
```
make run NP=<num processes> MATRIX_SIZE=<matrix_size> 
```
Compile and run predefined sized matrices
```
make small NP=<num processes> // 512x512
make medium NP=<num processes> // 2048x2048
make large NP=<num processes> // 16384x16384
make extralarge NP=<num processes> // 65536x65536
```

## Running on the Supercomputer

If you compiled manually do
```
srun ./matmul <matrix_size>
```
To compile and run, do not set NP, it is inferred from the supercomputer environment you are in.
set MPI_LAUNCH="srun"
```
make run MPI_LAUNCH="srun" MATRIX_SIZE=<matrix_size>
make small MPI_LAUNCH="srun"
make medium ...
```