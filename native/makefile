CUDA = /usr/local/cuda-5.5
NVCC = $(CUDA)/bin/nvcc
COMPUTCAPA = 30
GENCODE = -gencode arch=compute_$(COMPUTCAPA),code=sm_$(COMPUTCAPA)
CPPFLAGS = -O3 --compile -x c++
CUFLAGS = -O3 --compile $(GENCODE) -x cu

%.o:%.cpp
	$(NVCC) $(CPPFLAGS) $< -o $@
%.o:%.cu
	$(NVCC) $(CUFLAGS) $< -o $@

washall-cuda:adjmatrix.o cukernels.o main.o
	$(NVCC) --cudart static -link $^ -o $@
adjmatrix.o:adjmatrix.cpp
cukernels.o:cukernels.cu 
main.o:main.cpp
	$(NVCC) --compile -x c++ $< -o $@

clean:
	rm -f *.o washall-cuda
