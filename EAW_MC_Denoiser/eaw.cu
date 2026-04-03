#include <iostream>
#include<fstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cmath>

__constant__ float kernel[25];
__constant__ int2 offset[25];

__global__ void eaw_filter(
    const float4* colorMap,
    const float4* normalMap,
    const float4* positionMap,
    int stepwidth,
    float color_phi,
    float normal_phi,
    float position_phi,
    float4* outputMap,
    int width, 
    int height)
    
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y>=height){
        return;
    }

    int pixel = y * width + x;
    int pixel_tmp, x_tmp, y_tmp;
    float dist2, weight;
    float cum_w = 0.0;
    float4 sum {0,0,0,0};
    float4 ctmp, ntmp, ptmp;

    float4 cval = colorMap[pixel];
    float4 nval = normalMap[pixel];
    float4 pval = positionMap[pixel];

    for (int i=0; i<25;i++){

        x_tmp = x + offset[i].x * stepwidth;
        y_tmp = y + offset[i].y * stepwidth;

        if (x_tmp < 0) 
        {
            x_tmp = 0; }

        else if (x_tmp >= width) 
        {
            x_tmp = width - 1; }
        
        if (y_tmp < 0) 
        {
            y_tmp = 0; }

        else if (y_tmp >= height) 
        {
            y_tmp = height - 1; }

        pixel_tmp = y_tmp * width + x_tmp;

        ctmp = colorMap[pixel_tmp];
        dist2 = (cval.x - ctmp.x)*(cval.x - ctmp.x) + (cval.y - ctmp.y)*(cval.y - ctmp.y) + (cval.z - ctmp.z)*(cval.z - ctmp.z) + (cval.w - ctmp.w)*(cval.w - ctmp.w); 
        float c_w = fminf(expf(-(dist2)/color_phi), 1.0f);

        ntmp = normalMap[pixel_tmp];
        dist2 = fmaxf (((nval.x - ntmp.x)*(nval.x - ntmp.x) + (nval.y - ntmp.y)*(nval.y - ntmp.y) + (nval.z - ntmp.z)*(nval.z - ntmp.z) + (nval.w - ntmp.w)*(nval.w - ntmp.w))/(stepwidth*stepwidth), 0.0f);
        float n_w = fminf(expf(-(dist2)/normal_phi),1.0f);
        
        ptmp = positionMap[pixel_tmp];
        dist2 = (pval.x - ptmp.x)*(pval.x - ptmp.x) + (pval.y - ptmp.y)*(pval.y - ptmp.y) + (pval.z - ptmp.z)*(pval.z-ptmp.z) + (pval.w -ptmp.w)*(pval.w-ptmp.w);
        float p_w = fminf(expf(-(dist2)/position_phi),1.0f);
        
        weight = c_w * n_w * p_w;
        sum.x = sum.x + ctmp.x * weight * kernel[i];
        sum.y = sum.y + ctmp.y * weight * kernel[i];
        sum.z = sum.z + ctmp.z * weight * kernel[i];
        sum.w = sum.w + ctmp.w * weight * kernel[i];
        cum_w = cum_w + weight * kernel[i];
    }

    if (cum_w > 1e-8f) {
        outputMap[pixel] = make_float4(sum.x/cum_w, sum.y/cum_w, sum.z/cum_w, sum.w/cum_w);
    }
    else{
        outputMap[pixel] = colorMap[pixel];
    }
}


int main (void){
    int width = 512;
    int height = 512;
    int total_pixels = width*height;
    size_t pixel_size = sizeof(float4);
    size_t total_size = total_pixels * pixel_size;


    float4* colorMap;
    float4* normalMap;
    float4* positionMap;
    float4* outputMap;

    int stepwidth;
    float color_phi = 8.0f; 
    float normal_phi = 0.1f; 
    float position_phi = 80.0f;

    std::vector<float4> inputColorMap (total_pixels);
    std::vector<float4> inputNormalMap (total_pixels);
    std::vector<float4> inputPositionMap (total_pixels);
    std::vector<float4> outputFinalMap(total_pixels);

    std::ifstream fcolor("color.bin", std::ios::binary);
    fcolor.read(reinterpret_cast<char*>(inputColorMap.data()), total_size);
    fcolor.close();

    std::ifstream fnormal("normal.bin", std::ios::binary);
    fnormal.read(reinterpret_cast<char*>(inputNormalMap.data()), total_size);
    fnormal.close();

    std::ifstream fposition("position.bin", std::ios::binary);
    fposition.read(reinterpret_cast<char*>(inputPositionMap.data()), total_size);
    fposition.close();

    int2 inputOffset [25];

    float inputKernel [25] = { 0.0039, 0.0156,  0.0234,   0.0156,   0.0039,
                    0.0156,   0.0625,   0.0938,   0.0625,   0.0156,
                    0.0234,   0.0938,   0.1406,   0.0938,   0.0234,
                    0.0156,   0.0625,   0.0938,   0.0625,   0.0156,
                    0.0039,   0.0156,   0.0234,   0.0156,   0.0039};

    int count = 0;
    for (int i=-2; i<3; i++){
        for (int j=-2; j<3; j++){
            inputOffset[count] = make_int2(j,i);
            count = count + 1;
        }
    }

    cudaMalloc (&colorMap, total_size);
    cudaMalloc (&normalMap, total_size);
    cudaMalloc (&positionMap, total_size);
    cudaMalloc (&outputMap, total_size);

    cudaMemcpy (colorMap, inputColorMap.data(), total_size, cudaMemcpyHostToDevice);
    cudaMemcpy (normalMap, inputNormalMap.data(), total_size, cudaMemcpyHostToDevice);
    cudaMemcpy (positionMap, inputPositionMap.data(), total_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(kernel, inputKernel, sizeof(inputKernel));
    cudaMemcpyToSymbol(offset, inputOffset, sizeof(inputOffset));

    dim3 block_size(16, 16);
    dim3 grid_size(32, 32);

    for (int i=0; i<5; i++){
        stepwidth = static_cast<int>(pow(2,i));
        eaw_filter<<<grid_size, block_size>>>(
            colorMap,
            normalMap,
            positionMap,
            stepwidth,
            color_phi,
            normal_phi,
            position_phi,
            outputMap,
            width, 
            height);

        std::swap (colorMap, outputMap);
    
    }

    cudaMemcpy (outputFinalMap.data(), colorMap, total_size, cudaMemcpyDeviceToHost);
    std::ofstream fout("output.bin", std::ios::binary);
    fout.write(reinterpret_cast<const char*>(outputFinalMap.data()), width * height * sizeof(float4));
    fout.close();

    cudaFree (colorMap);
    cudaFree (normalMap);
    cudaFree (positionMap);
    cudaFree (outputMap);

    return 0;
}
