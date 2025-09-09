# Workshop 0: Intro to GT PACE

> Hi everyone! Welcome to Supercomputing@GT's first workshop of the semester, Intro to GT PACE! We are sure that you will learn all that you need to know about navigating GT's supercomputing cluster!

## What is Supercomputing@GT?

As one of the nation's most well-renowned research universities in high-performance computing (HPC), Supercomputing@GT seeks to educate Georgia Tech students on HPC's immense presence in research and industry alike. Today, we're hosting a workshop on GPU & CUDA fundamentals to help students learn how to utilize GPUs at Georgia Tech and elsewhere to develop the technologies of tomorrow.

## Today's Content

This workshop will primarily focus on what GT's supercomputing cluster is and how to navigate the system. Specifically, we will introduce you to using the Georgia Tech PACE Instructional Cluster (ICE-PACE). This repository contains all the files that are necessary for you to access throughout the workshop. All the files that will be used for hands-on practice with ICE-PACE is in the `/src` directory. A written version of the workshop material is in the `docs/main.pdf` file. Finally, the slides are in the `Workshop 0.pdf` file.

Now let us get the show on the road! :confetti_ball:
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
make large NP=<num processes> // 4096x4096
make extralarge NP=<num processes> // 8192x8192
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