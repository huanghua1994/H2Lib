LIB     = ../libh2.a
CC      = icc
CFLAGS  = -Wall -g -O3 -qopenmp -xHost -mkl -I../Library

EXE  = test_kernelmatrix.x
OBJS = kernelmatrix.o test_kernelmatrix.o $(LIB)

all: $(EXE)

%.o: %.c
	$(CC) ${CFLAGS} -I../ -c $^ 
	
test_kernelmatrix.x: $(OBJS)
	$(CC) ${CFLAGS} -o $@ $^ $(LIB)
	
clean:
	rm -f $(EXES)
