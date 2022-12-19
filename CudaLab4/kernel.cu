
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Место для вставки кода
__global__ void grayscale(float* in, float* out, int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h)
    {
        int grayOff = row * w + col;
        int rgbOff = grayOff * 3;
        float r = in[rgbOff];
        float g = in[rgbOff + 1];
        float b = in[rgbOff + 2];
        out[grayOff] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char* inputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* deviceInputImageData;
    float* deviceOutputImageData;

    args = wbArg_read(argc, argv); /* чтение входных аргументов */

    inputImageFile = wbArg_getInputFile(args, 0);

    inputImage = wbImport(inputImageFile);

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    // В данной лабораторной значение равно 3
    imageChannels = wbImage_getChannels(inputImage);

    //  Так как изображение монохромное, оно содержит только 1 канал
    outputImage = wbImage_new(imageWidth, imageHeight, 1);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    ///////////////////////////////////////////////////////
    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ Место для вставки кода
    dim3 blockSize(32, 32);
    dim3 grid(ceil(static_cast<float>(imageWidth) / blockSize.x), ceil(static_cast<float>(imageHeight) / blockSize.y));
    grayscale << <grid, blockSize >> > (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    ///////////////////////////////////////////////////////
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;

}
