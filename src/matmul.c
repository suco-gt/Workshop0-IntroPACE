/**
 * Basic Parallel Matrix Multiplication Program
 * Inspiration taken from https://github.com/zeilmannt/matrix-multiplication/
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> 
#include <mpi.h>

#define MAX_CONSOLE_MATRIX_SIZE 16
#define MAX_FILE_MATRIX_SIZE 256
#define OUTPUT_FILE "matrix_calculation.txt"

/**
 * generate_matrix
 * ---------------
 * Fills an NxN matrix with random float values in [start, end).
 *
 * Parameters:
 *   mat   - pointer to the float array to fill
 *   N     - size of the matrix (NxN)
 *   start - inclusive lower bound of the range
 *   end   - exclusive upper bound of the range
 *
 * Notes:
 *   Call srand() once before using this function to seed the RNG.
 */
void generate_matrix(float *mat, int N, float start, float end) {
    for (int i = 0; i < N * N; i++) {
        float r = (float)rand() / RAND_MAX;   // [0, 1)
        mat[i] = start + r * (end - start);   // [start, end)
    }
}

/**
 * get_matrix_string
 * -----------------
 * Converts an NxN matrix into a formatted string with aligned columns.
 *
 * Parameters:
 *   title - label for the matrix (e.g., "Matrix A")
 *   mat   - pointer to the float array (row-major order)
 *   N     - size of the matrix (NxN)
 *
 * Returns:
 *   Pointer to a heap-allocated string containing the formatted matrix.
 *   The caller is responsible for freeing the returned string.
 *
 * Notes:
 *   - Finds the widest element to align all columns properly.
 *   - Adds the title and newline characters for readability.
 */
char* get_matrix_string(const char *title, float *mat, int N) {
    int max_width = 0;
    char buffer[64];

    // First pass: find widest element
    for (int i = 0; i < N * N; i++) {
        int len = snprintf(buffer, sizeof(buffer), "%.3f", mat[i]);
        if (len > max_width) max_width = len;
    }

    // Estimate total size needed (rough estimate, may allocate extra)
    int estimated_size = N * N * (max_width + 4) + 1024;
    char *out = malloc(estimated_size);
    if (!out) return NULL;
    out[0] = '\0';

    // Add title
    strcat(out, title);
    strcat(out, ":\n");

    // Second pass: append each element
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            char line[64];
            snprintf(line, sizeof(line), "%*.*f ", max_width, 3, mat[i * N + j]);
            strcat(out, line);
        }
        strcat(out, "\n");
    }

    return out;
}

/**
 * print_matrix
 * ------------
 * Prints an NxN matrix to stdout with nicely aligned columns.
 *
 * Parameters:
 *   title - label for the matrix (e.g., "Matrix A")
 *   mat   - pointer to the float array (row-major order)
 *   N     - size of the matrix (NxN)
 *
 * Notes:
 *   - Internally calls get_matrix_string to format the matrix.
 *   - Frees the temporary string after printing.
 */
void print_matrix(const char *title, float *mat, int N) {
    char *matrix_str = get_matrix_string(title, mat, N);
    if (matrix_str) {
        printf("%s", matrix_str);
        free(matrix_str);
    }
}

/**
 * main
 * ----
 * Entry point of the MPI-based parallel matrix multiplication program.
 *
 * Responsibilities:
 *   - Initialize MPI environment.
 *   - Parse command-line arguments for matrix size.
 *   - Allocate memory for matrices (A, B, C) and local chunks.
 *   - Generate random matrices on rank 0.
 *   - Broadcast matrix B to all processes.
 *   - Scatter matrix A across processes.
 *   - Perform local matrix multiplication.
 *   - Gather local C chunks to rank 0.
 *   - Print matrices to console if N <= MAX_CONSOLE_MATRIX_SIZE.
 *   - Write matrices and execution info to OUTPUT_FILE.
 *   - Free allocated memory.
 *   - Finalize MPI.
 *
 * Usage:
 *   mpirun -np <num_processes> ./matrix_mpi <matrix_size>
 *
 * Returns:
 *   0 on success, non-zero on error (e.g., invalid arguments or memory allocation failure).
 */
int main(int argc, char* argv[]) {
    // srand(time(NULL)); // set the seed randomly every time the program is run
    srand(42); // fixed seed

    int rank, size, N;

    // Every MPI program requires you to initialize MPI through MPI_Init first
    MPI_Init(&argc, &argv);
    // The default communicator is comm world, this represents all of our processes
    // we can figure out the rank of our current process within the communicator, in a world using MPI_Comm_rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Similarly, we can get the total number of processes in our communicator
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // check for valid arguments
    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        // Always finalize mpi before exiting the program
        MPI_Finalize();
        return 1;
    }

    // atoi returns 0 if the input is not a valid integer
    // this is fine for the case where 0 is actually inputed since we don't want a 0x0 matrix
    N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size: must be a positive integer.\n");
        MPI_Finalize();
        return 1;
    }
    // we must be able to give equal sized chunks to each processor
    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size: must be divisible by number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    // how many rows of the matrix each process handles
    int rows_per_process = N / size;

    // Each process contains the entire, B, and a chunk of A, and C
    float *A = NULL, *B = NULL, *C = NULL;
    float *local_A = malloc(rows_per_process * N * sizeof(float));
    float *local_C = malloc(rows_per_process * N * sizeof(float));
    B = malloc(N * N * sizeof(float));

    if (!B || !local_A || !local_C) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = malloc(N * N * sizeof(float));
        // initialize C to all zeros
        C = calloc(N * N, sizeof(float));
        if (!A || !C) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // C is already set to 0's, randomly generate the A, B matrices
        generate_matrix(A, N, -100, 101);
        generate_matrix(B, N, -100, 101);
    }

    MPI_Barrier(MPI_COMM_WORLD); // ensure all processes start together

    if (rank == 0) {
        printf("Starting matrix multiplication with %d processes...\n", size);
    }
    // begin timer
    double start = MPI_Wtime();

    // int MPI_Bcast(
    //     void *buffer,           starting address of buffer to broadcast
    //     int count,              number of elements in buffer
    //     MPI_Datatype datatype,  type of each element
    //     int root,               rank of broadcasting process
    //     MPI_Comm comm,          communicator
    // );

    // this gives each process the entire B matrix
    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // int MPI_Scatter(
    //     const void *sendbuf,    starting address of send buffer (root only)
    //     int sendcount,          number of elements sent to each process
    //     MPI_Datatype sendtype,  type of each send element
    //     void *recvbuf,          starting address of receive buffer
    //     int recvcount,          number of elements received by each process
    //     MPI_Datatype recvtype,  type of each receive element
    //     int root,               rank of sending process
    //     MPI_Comm comm,          communicator
    // );

    // Spreads out A across all processes
    MPI_Scatter(A, rows_per_process * N, MPI_FLOAT,
                local_A, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Note we do not need to send C anywhere, since we initialized it to 0's

    // Local matrix multiplication
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // int MPI_Gather(
    //     const void *sendbuf,    starting address of local data to send
    //     int sendcount,          number of elements sent by each process
    //     MPI_Datatype sendtype,  type of each element sent
    //     void *recvbuf,          starting address of buffer to receive gathered data (root only)
    //     int recvcount,          number of elements received from each process
    //     MPI_Datatype recvtype,  type of each received element
    //     int root,               rank of receiving process
    //     MPI_Comm comm,          communicator
    // );

    // Gather the local C buffers to compile the entire C result matrix in one process
    MPI_Gather(local_C, rows_per_process * N, MPI_FLOAT,
               C, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); // ensure all processes end together
    double end = MPI_Wtime();   
    if (rank == 0) {
        printf("Finished Multiplication.\n");
    }

    if (rank == 0) {
        printf("Execution Time: %f seconds\nMatrix Size: %dx%d\nNumber of Processes: %d\n\n", end - start, N, N, size);

        if (N <= MAX_FILE_MATRIX_SIZE) {
            char *A_str = get_matrix_string("Matrix A", A, N);
            char *B_str = get_matrix_string("Matrix B", B, N);
            char *C_str = get_matrix_string("Matrix C", C, N);

            // Print to the console if the matrix is small enough
            if (N <= MAX_CONSOLE_MATRIX_SIZE) {
                printf("%s\n%s\n%s", A_str, B_str, C_str);
            } 

            FILE *f = fopen(OUTPUT_FILE, "w");
            if (f) {
                fprintf(f, "Execution Time: %f seconds\nMatrix Size: %dx%d\nNumber of Processes: %d\n\n", end - start, N, N, size);
                fprintf(f, "%s\n%s\n%s\n", A_str, B_str, C_str);
                fclose(f);
            } else {
                fprintf(stderr, "Failed to open file for writing\n");
            }

            free(A_str); 
            free(B_str); 
            free(C_str);
        }

        // free the large matrices only allocated on rank 0
        free(A); 
        free(C);
    }

    // All ranks have the entire B matrix and local parts of A and C
    free(B); free(local_A); free(local_C);

    // All MPI programs end with finalizing the MPI environment
    MPI_Finalize();

    return 0;
}