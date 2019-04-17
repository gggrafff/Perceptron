#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>
#include <cmath>
#include <string>
#include <fstream>

#define tr_speed 0.02f
#define block 256
//#define DEBUG
//#define SAFETY
#define TIMECONTROL
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX(i,j,ld) (((i)*(ld))+(j))
using namespace std;

__constant__ int dev_m;
__constant__ int dev_n;

#ifdef DEBUG
const char* cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}

const char* cudaGetErrorString(cudaError_t status)
{
	switch (status)
	{
	case cudaSuccess: return "cudaSuccess";
	case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
	case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
	case cudaErrorInitializationError: return "cudaErrorInitializationError";
	case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
	case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
	case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
	case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
	case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
	case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
	case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
	case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
	case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
	case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
	case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
	case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
	case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
	case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
	case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";
	case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";
	case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";
	case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
	case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";
	case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";
	case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";
	case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";
	case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";
	case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";
	case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";
	case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
	case cudaErrorUnknown: return "cudaErrorUnknown";
	case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";
	case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";
	case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
	case cudaErrorNotReady: return "cudaErrorNotReady";
	case cudaErrorStartupFailure: return "cudaErrorStartupFailure";
	case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";
	}
	return "unknown error";
}
#endif

__global__ void thresholdKernel(const float *in, float *out)
{
	__shared__ float temp[block];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < dev_n) {
		temp[threadIdx.x] = in[i];
	}
	//—инхронизируем все нити в блоке
	__syncthreads();
	if (i < dev_n) {
		//линейна€ функци€
		//out[i] = temp[threadIdx.x];

		//—игмоида
		out[i] = 1.0f / (1.0f + __expf(-temp[threadIdx.x]));
	}
}

__global__ void derivativeKernel(const float *in, float *out)
{
	__shared__ float temp[block];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < dev_n) {
		temp[threadIdx.x] = in[i];
	}
	//—инхронизируем все нити в блоке
	__syncthreads();
	if (i < dev_n) {
		//линейна€ функци€
		//out[i] = 1;

		//—игмоиды
		out[i] = __expf(temp[threadIdx.x]) / __powf((1.0f + __expf(temp[threadIdx.x])), 2);
	}
}

__global__ void productKernel(const float *in1, const float *in2, float *out)
{
	__shared__ float temp1[block];
	__shared__ float temp2[block];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < dev_n) {
		temp1[threadIdx.x] = in1[i];
		temp2[threadIdx.x] = in2[i];
	}
	//—инхронизируем все нити в блоке
	__syncthreads();
	if (i < dev_n) {
		out[i] = (float)dev_m*temp1[threadIdx.x] * temp2[threadIdx.x];
	}
}

class net {
private:
	cudaError_t cudaStatus;
	cublasStatus_t stat;
	cublasHandle_t handle;
public:
	net(int in_number, int layers_number, int* neurons_number) {
		this->layers_number = layers_number;
		this->in_number = in_number;
		this->neurons_number = new int[layers_number];
		for (int i = 0;i < layers_number;i++) this->neurons_number[i] = neurons_number[i];
		weights_number = in_number*neurons_number[0];
		for (int i = 1;i < layers_number;i++) weights_number += neurons_number[i] * neurons_number[i-1];
		weights = (float*)malloc(sizeof(int)*weights_number);
		for (int i = 0;i < weights_number;i++) weights[i] = (float)(rand() % 200 - 100) / 1000.0f;
		init();
	}
	net(string filename, bool txt_mode) {
		open(filename, txt_mode);
		init();
	}
	net(int in_number, int layers_number, int* neurons_number, float* weights) {
		this->layers_number = layers_number;
		this->in_number = in_number;
		this->neurons_number = new int[layers_number];
		for (int i = 0;i < layers_number;i++) this->neurons_number[i] = neurons_number[i];
		weights_number = in_number*neurons_number[0];
		for (int i = 1;i < layers_number;i++) weights_number += neurons_number[i] * neurons_number[i - 1];
		this->weights = (float*)malloc(sizeof(float)*weights_number);
		for (int i = 0;i < weights_number;i++) this->weights[i] = weights[i];
		init();
	}
	net& operator=(const net& right) {
		//проверка на самоприсваивание
		if (this == &right) {
			return *this;
		}
		this->layers_number = right.layers_number;
		this->in_number = right.in_number;
		this->neurons_number = new int[layers_number];
		for (int i = 0;i < layers_number;i++) this->neurons_number[i] = right.neurons_number[i];
		this->weights_number = right.weights_number;
		this->weights = (float*)malloc(sizeof(int)*weights_number);
		for (int i = 0;i < weights_number;i++) this->weights[i] = right.weights[i];
		init();
		return *this;
	}
	net(const net & object)
	{
		this->layers_number = object.layers_number;
		this->in_number = object.in_number;
		this->neurons_number = new int[layers_number];
		for (int i = 0;i < layers_number;i++) this->neurons_number[i] = object.neurons_number[i];
		this->weights_number = object.weights_number;
		this->weights = (float*)malloc(sizeof(float)*weights_number);
		for (int i = 0;i < weights_number;i++) this->weights[i] = object.weights[i];
		init();
	}
	void init() {
		cublasCreate(&handle);
		int maxnumin = in_number;
		for (int i = 0;i < layers_number;i++) if (maxnumin < neurons_number[i]) maxnumin = neurons_number[i];
		cudaMalloc((void**)&dev_one, maxnumin * sizeof(float));
		cudaMalloc((void**)&dev_two, maxnumin * sizeof(float));
		cudaMalloc((void**)&dev_weights, weights_number * sizeof(float));
		cudaMemcpy(dev_weights, weights, weights_number * sizeof(float), cudaMemcpyHostToDevice);
		for (int i = 0; i < layers_number; i++) neurnum += neurons_number[i];
		cudaMalloc((void**)&dev_net, neurnum * sizeof(float));
		cudaMalloc((void**)&dev_der, neurnum * sizeof(float));
		cudaMalloc((void**)&dev_y, neurnum * sizeof(float));
		cudaMalloc((void**)&dev_g, neurnum * sizeof(float));
		cudaMalloc((void**)&dev_onein, in_number * sizeof(float));
		cudaMalloc((void**)&dev_oneout, neurons_number[layers_number - 1] * sizeof(float));
	}
	void open(string filename, bool txt_mode) {
		if (txt_mode) {
			ifstream file(filename);
			file >> this->layers_number;
			file >> this->in_number;
			neurons_number = new int[this->layers_number];
			for (int i = 0;i < layers_number;i++) file >> neurons_number[i];
			weights_number = in_number*neurons_number[0];
			for (int i = 1;i < layers_number;i++) weights_number += neurons_number[i] * neurons_number[i - 1];
			this->weights = (float*)malloc(sizeof(float)*weights_number);
			for (int i = 0;i < weights_number;i++) file >> this->weights[i];
			file.close();
		}
		else {
			ifstream file(filename, ios::binary);
			file.read((char*)&(this->layers_number), sizeof(this->layers_number));
			file.read((char*)&(this->in_number), sizeof(this->in_number));
			neurons_number = new int[this->layers_number];
			file.read((char*)neurons_number, sizeof(int)*layers_number);
			weights_number = in_number*neurons_number[0];
			for (int i = 1;i < layers_number;i++) weights_number += neurons_number[i] * neurons_number[i - 1];
			this->weights = (float*)malloc(sizeof(float)*weights_number);
			file.read((char*)weights, sizeof(float)*weights_number);
			file.close();
		}
		init();
	}
	void save(string filename, bool txt_mode) {
		cudaMemcpy(weights, dev_weights, weights_number * sizeof(float), cudaMemcpyDeviceToHost);
		if (txt_mode) {
			ofstream file(filename);
			file << this->layers_number << " ";
			file << this->in_number << " ";
			for (int i = 0;i < layers_number;i++) file << neurons_number[i] << " ";
			for (int i = 0;i < weights_number;i++) file << this->weights[i] << " ";
			file.close();
		}
		else {
			ofstream file(filename, ios::binary);
			file.write((char*)&(this->layers_number), sizeof(this->layers_number));
			file.write((char*)&(this->in_number), sizeof(this->in_number));
			file.write((char*)neurons_number, sizeof(int)*layers_number);
			weights_number = in_number*neurons_number[0];
			for (int i = 1;i < layers_number;i++) weights_number += neurons_number[i] * neurons_number[i - 1];
			file.write((char*)weights, sizeof(float)*weights_number);
			file.close();
		}
	}
	~net() {
		delete weights;
		delete neurons_number;
		cudaFree(dev_der);
		cudaFree(dev_g);
		cudaFree(dev_net);
		cudaFree(dev_one);
		cudaFree(dev_onein);
		cudaFree(dev_oneout);
		cudaFree(dev_two);
		cudaFree(dev_weights);
		cudaFree(dev_y);
		cublasDestroy(handle);
	}

	cudaError_t thresholdWithCuda(float* in, float* out, int n) {
#ifdef DEBUG
		ofstream file;
		file.open("thresholdinctrl.txt");
		for (int i = 0;i < n;i++) {
			file << in[i] << " ";
		}
		file.close();
#endif
		float *dev_in = 0;
		float *dev_out = 0;
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
#ifdef DEBUG
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n";
			goto Error;
		}
#endif
		// Allocate GPU buffers for three vectors (one input, one output, one weights)    .
		cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(float));
#ifdef DEBUG
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMalloc failed!\n";
			goto Error;
		}
#endif
		cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(float));
#ifdef DEBUG
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMalloc failed!\n";
			goto Error;
		}
#endif
		// Copy vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(float), cudaMemcpyHostToDevice);
#ifdef DEBUG
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMemcpy failed!\n";
			goto Error;
		}
#endif
		cudaStatus = cudaMemcpyToSymbol(dev_n, &n, sizeof(int), 0, cudaMemcpyHostToDevice);
#ifdef DEBUG
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
		// Launch a kernel on the GPU with one thread for each element.
		thresholdKernel <<<ceil((float)n/(float)block), block >>>(dev_in, dev_out);
#ifdef DEBUG
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "thresholdKernel launch failed: "<< cudaGetErrorString(cudaStatus) <<"\n";
			goto Error;
		}
#endif
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
#ifdef DEBUG
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaDeviceSynchronize returned error code "<< cudaStatus <<" after launching thresholdKernel!\n";
			goto Error;
		}
#endif
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(out, dev_out, n * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef DEBUG
		cout << cudaGetErrorString(cudaStatus) << "\n";
#endif
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMemcpy failed!\n";
			goto Error;
		}
	Error:
#endif
		cudaFree(dev_in);
		cudaFree(dev_out);

#ifdef DEBUG
		file.open("thresholdoutctrl.txt");
		for (int i = 0;i < n;i++) {
			file << out[i] << " ";
		}
		file.close();
#endif

		return cudaStatus;
	}

	cudaError_t derivativeWithCuda(float* in, float* out, int n) {
#ifdef DEBUG
		ofstream file;
		file.open("derivativeinctrl.txt");
		for (int i = 0;i < n;i++) {
			file << in[i] << " ";
		}
		file.close();
#endif
		float *dev_in = 0;
		float *dev_out = 0;
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n";
			goto Error;
		}
#endif
		// Allocate GPU buffers for three vectors (one input, one output, one weights)    .
		cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(float));
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMalloc failed!\n";
			goto Error;
		}
#endif
		cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(float));
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMalloc failed!\n";
			goto Error;
		}
#endif
		// Copy vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(float), cudaMemcpyHostToDevice);
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMemcpy failed!\n";
			goto Error;
		}
#endif
		cudaMemcpyToSymbol(dev_n, &n, sizeof(int), 0, cudaMemcpyHostToDevice);

		// Launch a kernel on the GPU with one thread for each element.
		derivativeKernel <<<ceil((float)n / (float)block), block >>>(dev_in, dev_out);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "thresholdKernel launch failed: "<< cudaGetErrorString(cudaStatus) <<"\n";
			goto Error;
		}
#endif
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaDeviceSynchronize returned error code "<< cudaStatus <<" after launching thresholdKernel!\n";
			goto Error;
		}
#endif
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(out, dev_out, n * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef SAFETY
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMemcpy failed!\n";
			goto Error;
		}
	Error:
#endif
		cudaFree(dev_in);
		cudaFree(dev_out);

#ifdef DEBUG
		file.open("derivativeoutctrl.txt");
		for (int i = 0;i < n;i++) {
			file << out[i] << " ";
		}
		file.close();
#endif

		return cudaStatus;
	}

	int calcWithCuda(float *in, float *out)
	{

#if (defined TIMECONTROL) || (defined DEBUG)
		ofstream file;
#endif
#ifdef TIMECONTROL
		cudaEvent_t start, stop;
		float gpuTime = 0.0f;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
#endif
#ifdef DEBUG
		file.open("calcinctrl.txt");
		for (int i = 0;i < in_number;i++) {
			file << in[i] << " ";
		}
		file.close();
#endif

		cudaMemcpy(dev_one, in, in_number * sizeof(float), cudaMemcpyHostToDevice);
		float a = 1, b = 0;
		int wnum = 0;

		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			1, neurons_number[0], in_number,
			&a,
			dev_one, in_number,
			dev_weights + wnum, in_number,
			&b,
			dev_two, 1);
		wnum += in_number*neurons_number[0];
		cudaMemcpyToSymbol(dev_n, &in_number, sizeof(int), 0, cudaMemcpyHostToDevice);
		thresholdKernel << <ceil((float)in_number / (float)block), block >> >(dev_two, dev_one);
		cudaDeviceSynchronize();

		for (int l = 1; l < layers_number;l++) {
			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
				1, neurons_number[l], neurons_number[l - 1],
				&a,
				dev_one, neurons_number[l - 1],
				dev_weights + wnum, neurons_number[l - 1],
				&b,
				dev_two, 1);
			wnum += neurons_number[l - 1] *neurons_number[l];
			cudaMemcpyToSymbol(dev_n, &(neurons_number[l - 1]), sizeof(int), 0, cudaMemcpyHostToDevice);
			thresholdKernel << <ceil((float)neurons_number[l - 1] / (float)block), block >> >(dev_two, dev_one);
			cudaDeviceSynchronize();
		}
		cudaMemcpy(out, dev_one, neurons_number[layers_number-1] * sizeof(float), cudaMemcpyDeviceToHost);


#ifdef DEBUG
		file.open("calcoutctrl.txt");
		for (int i = 0;i < neurons_number[layers_number - 1];i++) {
			file << out[i] << " ";
		}
		file.close();
#endif
#ifdef TIMECONTROL
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpuTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		file.open("timecontrol.txt", ios::app);
		file << "calc => " << gpuTime << "ms\n";
		file.close();
#endif
		return 0;
	}

	int trainingWithCuda(float *in, float *out) {
#if (defined TIMECONTROL) || (defined DEBUG)
		ofstream file;
#endif
#ifdef TIMECONTROL
		cudaEvent_t start, stop;
		float gpuTime = 0.0f;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
#endif

		cudaMemcpy(dev_onein, in, in_number * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_oneout, out, neurons_number[layers_number - 1] * sizeof(float), cudaMemcpyHostToDevice);

		float a = 1, b = 0;
		int wnum = 0;
		int ynum = 0;
		int nnum = 0;
		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			1, neurons_number[0], in_number,
			&a,
			dev_onein, in_number,
			dev_weights + wnum, in_number,
			&b,
			dev_net+nnum, 1);
		wnum += in_number*neurons_number[0];
		cudaMemcpyToSymbol(dev_n, &in_number, sizeof(int), 0, cudaMemcpyHostToDevice);
		thresholdKernel << <ceil((float)in_number / (float)block), block >> >(dev_net + nnum, dev_y + ynum);
		cudaDeviceSynchronize();
		nnum += neurons_number[0];
		
		for (int l = 1; l < layers_number;l++) {
			cudaMemcpyToSymbol(dev_n, &neurons_number[l], sizeof(int), 0, cudaMemcpyHostToDevice);
			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
				1, neurons_number[l], neurons_number[l - 1],
				&a,
				dev_y+ynum, neurons_number[l - 1],
				dev_weights + wnum, neurons_number[l - 1],
				&b,
				dev_net +nnum, 1);
			ynum += neurons_number[l - 1];
			wnum += neurons_number[l - 1] * neurons_number[l];
			thresholdKernel << <ceil((float)neurons_number[l] / (float)block), block >> >(dev_net + nnum, dev_y + ynum);
			cudaDeviceSynchronize();
			nnum += neurons_number[l];
	}
		cudaMemcpyToSymbol(dev_n, &neurnum, sizeof(int), 0, cudaMemcpyHostToDevice);
		derivativeKernel << <ceil((float)neurnum / (float)block), block >> >(dev_net, dev_der);
//============================================
		wnum = weights_number;
		ynum = neurnum- neurons_number[layers_number - 1];
		int dnum = neurnum - neurons_number[layers_number - 1];
		int gnum = neurnum- neurons_number[layers_number - 1];
		float alpha = -1;
		cublasSaxpy(handle, neurons_number[layers_number-1],
			&alpha,
			dev_y+ynum, 1,
			dev_oneout, 1);
		cudaMemcpyToSymbol(dev_n, &neurons_number[layers_number - 1], sizeof(int), 0, cudaMemcpyHostToDevice);
		int m = 2;
		cudaMemcpyToSymbol(dev_m, &m, sizeof(int), 0, cudaMemcpyHostToDevice);
		productKernel << <ceil((float)neurons_number[layers_number - 1] / (float)block), block >> >(dev_oneout, dev_der + dnum,dev_g+gnum);
		cudaDeviceSynchronize();

		//nnum += neurons_number[0];
		//wnum -= in_number*neurons_number[0];
		m = 1;
		cudaMemcpyToSymbol(dev_m, &m, sizeof(int), 0, cudaMemcpyHostToDevice);
		for (int l = layers_number-2; l >=0 ;l--) {
			cudaMemcpyToSymbol(dev_n, &neurons_number[l], sizeof(int), 0, cudaMemcpyHostToDevice);
			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
				1, neurons_number[l], neurons_number[l + 1],
				&a,
				dev_g + gnum, neurons_number[l + 1],
				dev_weights + wnum, neurons_number[l],
				&b,
				dev_g + gnum- neurons_number[l], 1);
			gnum -= neurons_number[l];
			wnum -= neurons_number[l + 1] * neurons_number[l];
			dnum -= neurons_number[l];
			productKernel << <ceil((float)neurons_number[l] / (float)block), block >> >(dev_g + gnum, dev_der + dnum, dev_g + gnum);
			cudaDeviceSynchronize();
	}

//========================
		a = tr_speed;
		b = 1;
		gnum = 0;
		wnum = 0;
		ynum = 0;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			in_number, neurons_number[0], 1,
			&a,
			dev_onein, in_number,
			dev_g + nnum, 1,
			&b,
			dev_weights + wnum, in_number);
		gnum += neurons_number[0];
		wnum += in_number * neurons_number[0];
		for (int l = 1; l < layers_number;l++) {
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				neurons_number[l-1], neurons_number[l], 1,
				&a,
				dev_y+ynum, neurons_number[l - 1],
				dev_g + gnum, 1,
				&b,
				dev_weights + wnum, neurons_number[l - 1]);
			gnum += neurons_number[l];
			wnum += neurons_number[l - 1] * neurons_number[l];
			ynum += neurons_number[l - 1];
		}


#ifdef TIMECONTROL
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpuTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		file.open("timecontrol.txt", ios::app);
		file << "training => " << gpuTime << "ms\n";
		file.close();
#endif
		return 0;
	}

	int layers_number;
	int in_number;
	int weights_number;
	int neurnum;
	int* neurons_number;
	float* weights;
	float *dev_one;
	float *dev_two;
	float *dev_weights;
	float* dev_net;
	float* dev_der;
	float* dev_y;
	float* dev_g;
	float *dev_onein;
	float *dev_oneout;
};

void txt_read(float* in, int in_number, string filename) {
	ifstream file(filename);
	int i = 0;
	for (;i < in_number;i++) {
		if (file.eof()) break;
		file >> in[i];
	}
	file.close();
	for (;i < in_number;i++) in[i] = 0;
}
struct bmpinfo {
	int size = 0, pixels_adress = 0, width = 0, height = 0;
	short int bits_per_pixel = 0;
};
bmpinfo bmp_read(float* in, int in_number, string filename) {
	// ќткрываем файл
	ifstream file(filename, ios::in | ios::binary);

	bmpinfo info;

	// ѕереходим на 2 байт
	file.seekg(2, ios::beg);

	// —читываем размер файла
	file.read((char*)&info.size, sizeof(int));

	// ѕереходим на 10 байт
	file.seekg(10, ios::beg);

	// —читываем адрес начала массива пикселей
	file.read((char*)&info.pixels_adress, sizeof(int));

	file.seekg(18, ios::beg);

	// —читываем ширину и высоту изображени€ (в пиксел€х)
	file.read((char*)&info.width, sizeof(int));

	file.read((char*)&info.height, sizeof(int));


	file.seekg(28, ios::beg);

	// считываем кол-во битов на пиксель
	file.read((char*)&info.bits_per_pixel, sizeof(short int));

	file.seekg(info.pixels_adress, ios::beg);

	int i = 0;
	int offset = (32 - ((info.bits_per_pixel*info.width) % 32)) / 8;
	///////////////////// 1 BIT
	if (info.bits_per_pixel == 1)
	{
		unsigned char bgr;

		for (int y = info.height-1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 1);

				if (x >= info.width-4)
				{
					for (int n = 7; n >= 4; n--)
					{
						if (bgr & (1 << n)) // if 1 bit of readed pixels and 0b10000000
							in[i++] = 0;
						else
							in[i++] = 1;

						if (n != 4) x++;
					}
				}

				else
				{
					for (int n = 7; n >= 0; n--)
					{
						if (bgr & (1 << n)) // if 1 bit of readed pixels and 0b10000000
							in[i++] = 0;
						else
							in[i++] = 1;

						if (n != 0) x++;
					}
				}
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 1 bit END
	///////////////////// 4 BIT
	else if (info.bits_per_pixel == 4)
	{
		unsigned char bgr;

		for (int y = info.height-1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{

				file.read((char*)&bgr, 1);

				if (bgr & 0xF0)
					in[i++] = 0;
				else
					in[i++] = 1;

				x++;

				if (bgr & 0x0F)
					in[i++] = 0;
				else
					in[i++] = 1;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 4 bit END
	///////////////////// 8 BIT
	else if (info.bits_per_pixel == 8)
	{
		unsigned char bgr;

		for (int y = info.height-1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{

				file.read((char*)&bgr, 1);

				if (bgr == 0xFF)
					in[i++] = 0;
				else
					in[i++] = 1;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 8 bit END
	///////////////////// 16 BIT
	else if (info.bits_per_pixel == 16)
	{
		unsigned short int bgr;

		for (int y = info.height-1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 2);

				if (bgr >= 0xFFF)
					in[i++] = 0;
				else
					in[i++] = 1;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 16 bit END
	///////////////////// 24 BIT
	else if (info.bits_per_pixel == 24)
	{
		unsigned int bgr = 0;

		for (int y = info.height-1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 3);

				//cout << bgr << endl;

				if (bgr == 0xFFFFFF)
					in[i++] = 0;
				else
					in[i++] = 1;

				bgr = 0;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 24 bit END
	///////////////////// 32 BIT
	else if (info.bits_per_pixel == 32)
	{
		unsigned int bgr;

		for (int y = info.height-1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 4);

				if (bgr >= 0xFFFFFF)
					in[i++] = 0;
				else
					in[i++] = 0;
			}
		}
	}
	//////////////// 32 bit END
	else
	{
		cerr << "»звините, ¬аше изображение должно иметь 1, 4, 8, 16, 24 или 32 бит на пиксель. " << endl;
	}
	file.close();
	for (;i < in_number;i++) in[i] = 0;
	return info;
}



int main()
{
	srand(time(0)); // автоматическа€ рандомизаци€

	string comand="";
	bool training_mode = false;
	bool bmp_mode = false;
	bool txt_mode = false;
	string infile="in.txt";
	string outfile = "out.txt";
	setlocale(LC_ALL, "Russian");
	cout << "¬ерси€ дл€ CUDA. Ѕыстра€. \n¬ведите команду:\n\"file\" - загрузить нейросеть из файла\n\"txt\" - установить тип входного файла - текстовый\n\"bin\" - установить тип входного файла - двоичный\n\"new\" - создать новую нейросеть\n\"exit\" - выйти\n";
	net* neuro;
	while (comand != "exit") {
		cin >> comand;
#ifdef DEBUG
		cout <<"¬ведено в цикле 1: "<< comand<<"\n";
#endif
		if (comand == "exit") break;
		if (comand == "file") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro = new net(file, txt_mode);
			cout << "OK\n";
			break;
		}
		if (comand == "txt") {
			txt_mode = true;
		}
		if (comand == "bin") {
			txt_mode = false;
		}
		if (comand == "new") {
			int layers_number;
			cout << "¬ведите количество слоЄв:\n";
			cin >> layers_number;
			int in_number;
			cout << "¬ведите количество входов сети:\n";
			cin >> in_number;
			int* neurons_number = new int[layers_number];
			for (int i = 0;i < layers_number;i++) {
				cout << "¬ведите количество нейронов "<< i <<"-го сло€:\n";
				cin >> neurons_number[i];
			}
			neuro = new net(in_number, layers_number, neurons_number);
			cout << "OK\n";
			break;
		}
	}
	cout << "¬ведите команду:\n\"normal\" - перейти в рабочий режим (по умолчанию)\n\"training\" - перейти в режим обучени€\n";
	cout << "\"txt\" - указать входной текстовый файл (по умолчанию in.txt)\n\"bmp\" - указать входной *.bmp файл\n";
	cout << "\"out\" - указать выходной текстовый файл (по умолчанию out.txt)\n";
	cout << "\"run\" - запустить\n\"save\" - сохранить сеть в файл\n";
	cout << "\"open\" - загрузить сеть из файла\n\"exit\" - выйти\n";
	while (comand != "exit") {
		cin >> comand;
#ifdef DEBUG
		cout << "¬ведено в цикле 2: " << comand << "\n";
#endif
		if (comand == "exit") break;
		if (comand == "normal") training_mode = false;
		if (comand == "training") training_mode = true;
		if (comand == "txt") {
			bmp_mode = false;
			cout << "¬ведите адрес файла:\n";
			cin >> infile;
#ifdef DEBUG
			cout << "¬ведено: " << infile << "\n";
#endif
			cout << "OK\n";
		}
		if (comand == "bmp") {
			bmp_mode = true;
			cout << "¬ведите адрес файла:\n";
			cin >> infile;
#ifdef DEBUG
			cout << "¬ведено: " << infile << "\n";
#endif
			cout << "OK\n";
		}
		if (comand == "out") {
			cout << "¬ведите адрес файла:\n";
			cin >> outfile;
#ifdef DEBUG
			cout << "¬ведено: " << outfile << "\n";
#endif
			cout << "OK\n";
		}
		if (comand == "save") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro->save(file, txt_mode);
			cout << "OK\n";
		}
		if (comand == "open") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro->open(file, txt_mode);
			cout << "OK\n";
		}
		if (comand == "run") {
			float* in=new float[neuro->in_number];
			if (!bmp_mode) {
				txt_read(in, neuro->in_number, infile);	
			}
			else {
				bmpinfo info=bmp_read(in, neuro->in_number, infile);
#ifdef DEBUG
				cout << "Bits per pixel: "<<info.bits_per_pixel<<"\n";
				cout << "Height: " << info.height << "\n";
				cout << "Width: " << info.width << "\n";
				cout << "Pixels adress: " << info.pixels_adress << "\n";
				cout << "Size: " << info.size << "\n";
#endif
			}
#ifdef DEBUG
			ofstream file("inctrl.txt");
			for (int i = 0;i < neuro->in_number;i++) {
				file << in[i] << " ";
			}
			file.close();
#endif
			if (!training_mode) {
				float* out = (float*)malloc(neuro->neurons_number[neuro->layers_number-1]*sizeof(float));
				neuro->calcWithCuda(in, out);
				ofstream file(outfile);
				for (int i = 0;i < neuro->neurons_number[neuro->layers_number - 1];i++) {
					file << out[i] << " ";
				}
				file.close();
			}else{
				ifstream file(outfile);
				float* out = new float[neuro->neurons_number[neuro->layers_number - 1]];
				int i = 0;
				for (;i < neuro->neurons_number[neuro->layers_number - 1];i++) {
					if (file.eof()) break;
					file >> out[i];
				}
				file.close();
				for (;i < neuro->neurons_number[neuro->layers_number - 1];i++) out[i] = 0;

#ifdef DEBUG
				ofstream file2("outctrl.txt");
				for (int i = 0;i < neuro->neurons_number[neuro->layers_number - 1];i++) {
					file2 << out[i] << " ";
				}
				file2.close();
#endif
				neuro->trainingWithCuda(in, out);
			}
			cout << "OK\n";
		}
	}
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaDeviceReset failed!\n";
		return 1;
	}
    return 0;
}

