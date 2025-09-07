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

ifeq ($(MPI_LAUNCH),srun)
  MPI_OPTS ?=
else
  MPI_OPTS ?= -np $(NP)
endif

# Default target
all: clean $(TARGET)

# Build executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

define RUN
@if [ "$(MPI_LAUNCH)" = "srun" ]; then \
	$(MPI_LAUNCH) ./$(TARGET) $(MATRIX_SIZE); \
else \
	$(MPI_LAUNCH) -np $(NP) ./$(TARGET) $(MATRIX_SIZE); \
fi
endef

# Run rules
run: $(TARGET)
	$(RUN)

small: MATRIX_SIZE = 512
small: $(TARGET)
	$(RUN)

medium: MATRIX_SIZE = 2048
medium: $(TARGET)
	$(RUN)

large: MATRIX_SIZE = 4096
large: $(TARGET)
	$(RUN)

extralarge: MATRIX_SIZE = 8192
extralarge: $(TARGET)
	$(RUN)

# Clean up
clean:
	rm -f $(TARGET) $(OUTPUT_FILE)

.PHONY: all run clean small medium large extralarge
