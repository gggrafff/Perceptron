
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{
	int deviceCount;
	cudaDeviceProp devProp;
	cudaGetDeviceCount(&deviceCount);

	printf("Found %d devices\n", deviceCount);

	for (int device=0; device < deviceCount; device++)
	{
		cudaGetDeviceProperties(&devProp, device);

		printf("Device %d\n", device);
		printf("Compute capability: %d.%d\n", devProp.major, devProp.minor);
		printf("Name: %s\n", devProp.name);
		printf("Total Global Memory: %d\n", devProp.totalGlobalMem);
		printf("Shared memory per block: %d\n", devProp.sharedMemPerBlock);
		printf("Registers per block: %d\n", devProp.regsPerBlock);
		printf("Warp size: %d\n", devProp.warpSize);
		printf("Max threads per block: %d\n", devProp.maxThreadsPerBlock);
		printf("Total constant memory: %d\n", devProp.totalConstMem);
		printf("Clock Rate: %d\n", devProp.clockRate);
		printf("Texture Alignment: %u\n", devProp.textureAlignment);
		printf("Device overlap: %d\n", devProp.deviceOverlap);
		printf("Multiprocessot Count: %d\n", devProp.multiProcessorCount);
		printf("Max Threads Dim: %d %d %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
		printf("Max Grid Size: %d %d %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
	}
	getchar();
    return 0;
}
