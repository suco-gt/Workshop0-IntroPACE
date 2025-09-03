CC = mpicc            					# mpi compiler
CFLAGS = -Wall -O3    					# warning flags + optimization
TARGET = matmul       					# name of the executable
SRC = matmul.c        					# source file
NP ?= 1               					# default number of processes, can be overriden with NP=<value>
MATRIX_SIZE ?= $(shell expr $(NP) \* 4) # size of the square matrix   # size of the square matrix
OUTPUT_FILE = matrix_calculation.txt    # file that gets created with the A, B, and resultant C matrices

# By default make executes the first rule/target? what is this called
# so we begin with an all to catch that and say we require the Target executable
all: $(TARGET)

# Rule to build the executable, depends on the source file
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Run rule (depends on the executable existing)
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(MATRIX_SIZE)

# Clean up
clean:
	rm -f $(TARGET) $(OUTPUT_FILE)

small: MATRIX_SIZE = 512
small: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(MATRIX_SIZE)

medium: MATRIX_SIZE = 2048
medium: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(MATRIX_SIZE)

large: MATRIX_SIZE = 16384
large: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(MATRIX_SIZE)

extralarge: MATRIX_SIZE = 65536
extralarge: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(MATRIX_SIZE)

.PHONY: all run clean small medium large extralarge
