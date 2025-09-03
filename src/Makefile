# Compiler and flags
CC = mpicc
CFLAGS = -Wall -O3
TARGET = matmul
SRC = matmul.c

# Number of processes (default) and matrix size
NP ?= 1
MATRIX_SIZE ?= $(shell expr $(NP) \* 4)

# Output file
OUTPUT_FILE = matrix_calculation.txt

# MPI launcher and options (can be overridden for Slurm)
MPI_LAUNCH ?= mpirun
MPI_OPTS ?= -np $(NP)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Run rules
run: $(TARGET)
	$(MPI_LAUNCH) $(MPI_OPTS) ./$(TARGET) $(MATRIX_SIZE)

small: MATRIX_SIZE = 512
small: $(TARGET)
	$(MPI_LAUNCH) $(MPI_OPTS) ./$(TARGET) $(MATRIX_SIZE)

medium: MATRIX_SIZE = 2048
medium: $(TARGET)
	$(MPI_LAUNCH) $(MPI_OPTS) ./$(TARGET) $(MATRIX_SIZE)

large: MATRIX_SIZE = 16384
large: $(TARGET)
	$(MPI_LAUNCH) $(MPI_OPTS) ./$(TARGET) $(MATRIX_SIZE)

extralarge: MATRIX_SIZE = 65536
extralarge: $(TARGET)
	$(MPI_LAUNCH) $(MPI_OPTS) ./$(TARGET) $(MATRIX_SIZE)

# Clean up
clean:
	rm -f $(TARGET) $(OUTPUT_FILE)

.PHONY: all run clean small medium large extralarge
