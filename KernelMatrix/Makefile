LIB     = ../libh2.a
CC      = icc
CFLAGS  = -Wall -g -O3 -qopenmp -xHost -mkl -I../Library

EXES = test_kernelmatrix.x test_kernelmatrix_fin.x
OBJS = kernelmatrix.o test_kernelmatrix.o $(LIB)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS) -I../ -c $^ 
	
test_kernelmatrix.x: kernelmatrix.o test_kernelmatrix.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ 

test_kernelmatrix_fin.x: kernelmatrix.o test_kernelmatrix_fin.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ 
	
clean:
	rm -f $(EXES) *.o
