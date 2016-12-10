default: part1

part1: project3.cu
	nvcc -arch=sm_20 project3.cu -o part1
