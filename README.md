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

These commands are for running the program **on your own computer**.

### Compile the program

This creates an executable called `matmul`.

```bash
make
```

You can then run the executable using `mpirun`:

```bash
mpirun -n <num processes> ./matmul <matrix_size>
```

For example:

```bash
mpirun -n 4 ./matmul 512
```

This runs the program with 4 MPI processes on a 512 × 512 matrix.

### Compile and run in one command

You can also compile and run the program in one command.

* `NP` specifies the number of MPI processes.
* `MATRIX_SIZE` specifies the size of the matrix.
* If `NP` is not provided, it defaults to 1.
* If `MATRIX_SIZE` is not provided, it defaults to 4 × `NP`.

```bash
make run NP=<num processes> MATRIX_SIZE=<matrix_size>
```

For example:

```bash
make run NP=4 MATRIX_SIZE=512
```

### Compile and run predefined matrix sizes

You can use the following commands to run predefined matrix sizes:

```bash
make small NP=<num processes>       # 512 × 512
make medium NP=<num processes>      # 2048 × 2048
make large NP=<num processes>       # 4096 × 4096
make extralarge NP=<num processes> # 8192 × 8192
```

For example:

```bash
make medium NP=4
```

---

## Running on the Supercomputer

These commands are for running the program **on the PACE supercomputer**.

> **Important:** When running on the supercomputer, do **not** specify `NP`. The number of processes is automatically determined by the resources allocated to you.

### If you already compiled the program

If you have already created the `matmul` executable using `make`, you can run it with `srun`:

```bash
srun ./matmul <matrix_size>
```

For example:

```bash
srun ./matmul 512
```

### Compile and run in one command

You can also compile and run the program in one command.

When running on the supercomputer, set `MPI_LAUNCH` to `srun`:

```bash
make run MPI_LAUNCH="srun" MATRIX_SIZE=<matrix_size>
```

For example:

```bash
make run MPI_LAUNCH="srun" MATRIX_SIZE=512
```

**Do not include `NP` in this command.** The number of processes is inferred from the supercomputer environment.

### Compile and run predefined matrix sizes

You can also use the predefined matrix sizes:

```bash
make small MPI_LAUNCH="srun"       # 512 × 512
make medium MPI_LAUNCH="srun"      # 2048 × 2048
make large MPI_LAUNCH="srun"       # 4096 × 4096
make extralarge MPI_LAUNCH="srun"  # 8192 × 8192
```

For example:

```bash
make medium MPI_LAUNCH="srun"
```

**Remember:** Do not specify `NP` when running on the supercomputer.
