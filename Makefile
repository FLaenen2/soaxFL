CUDA=0
CPPVER=-std=c++11
OPTIM=-O3
OS = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
ifeq ($(OS),darwin)
    CXX = clang++
else
    CXX = /home/holger/gcc-4.8/bin/g++
endif
CC = nvcc
CXXFLAGS =

X_INC = -I/usr/lib/openmpi/include/ -I/softs/intel/impi/5.0.1.035/intel64/include

all: main 

#libsoax.a: soaxWriterMPI.o soaxWriterThreaded.o
OBJS = main.o
ifeq ($(CUDA),1)
    OBJS += gpuKernels.o 
endif


main:	$(OBJS)
	$(CC) $(CXXFLAGS) -o main $(OBJS)

%.o:	%.cu
	$(CC) $(OPTIM) $(CPPVER) -o $@ -c $< 
	
%.o:	%.cpp
	$(CC) $(OPTIM) $(CPPVER) -o $@ -c $< 

gpuKernels.o:	gpuKernels.cu
	$(NVCC) -gencode arch=compute_20,code=sm_20 -ccbin $(CXX) $(CPPVER) $(OPTIM) $(CXXFLAGS) $(X_INC) -c $<


.PHONY: clean
clean:
	rm -f *.~ *.o libsoax.a main
