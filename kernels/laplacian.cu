#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <nvtx3/nvToolsExt.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_SIZE 16
using unchar = unsigned char;
using int16 = short;

struct Saturate {
    __host__ __device__ void operator()(thrust::tuple<int16, unchar&> t) const {
        int16 val_in = thrust::get<0>(t);
        int val = (int)val_in;        
        if (val < 0) val = -val; 
        if (val > 255) val = 255;
        
        thrust::get<1>(t) = (unchar)val;
    }
};

__global__ void laplacian_kernel(const unchar* __restrict__ img_in, int16* img_out, int width, int height) {
    __shared__ unchar cache[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int sx = tx + 1; 
    int sy = ty + 1;

    auto get_clamped = [=](int x, int y) -> unchar{
        int cx = min(max(x,0), width-1);
        int cy = min(max(y,0), height-1);
        return img_in[cy*width + cx];
    };

    cache[sy][sx] = get_clamped(gx,gy);
    if (ty == 0) cache[0][sx] = get_clamped(gx, gy - 1);
    if (ty == BLOCK_SIZE - 1) cache[sy + 1][sx] = get_clamped(gx, gy + 1);
    if (tx == 0) cache[sy][0] = get_clamped(gx-1, gy);
    if (tx == BLOCK_SIZE - 1) cache[sy][sx + 1] = get_clamped(gx + 1,gy);

    __syncthreads();

    if (gx < width && gy < height) {
        //  0  1  0
        //  1 -4  1
        //  0  1  0
        int sum = 0;
        sum += cache[sy - 1][sx];     // g
        sum += cache[sy][sx - 1];     // l
        sum += cache[sy][sx] * (-4);  // śr
        sum += cache[sy + 1][sx];     // d
        sum += cache[sy][sx + 1];     // p
        
        img_out[gy * width + gx] = (int16)sum;
    }
}

torch::Tensor laplacian(torch::Tensor input) {
    nvtx3::scoped_range r{"Full_GPU_Pipeline_Extension"};

    int height = input.size(0);
    int width = input.size(1);
    int size_pixels = width * height;
    size_t size_bytes = size_pixels * sizeof(unchar);
    //cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    //auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    torch::Tensor output = torch::empty({height, width}, options);

    cudaHostRegister(input.data_ptr(),size_bytes,cudaHostRegisterDefault);
    cudaHostRegister(output.data_ptr(), size_bytes, cudaHostRegisterDefault);

    unchar *d_in,*d_out_final;
    int16 *d_raw;

    cudaMalloc(&d_in,size_bytes);
    cudaMalloc(&d_out_final,size_bytes);
    cudaMalloc(&d_raw, size_pixels * sizeof(int16));
    //cudaMallocAsync(&d_raw, size_pixels * sizeof(int16), stream);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_in,input.data_ptr(),size_bytes,cudaMemcpyHostToDevice, stream);
    //const unchar* d_in = input.data_ptr<unchar>();
    //unchar* d_out_final = output.data_ptr<unchar>();
    {
        nvtx3::scoped_range r_kern{"Kernel_Launch"};
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
        
        laplacian_kernel<<<blocks, threads, 0, stream>>>(d_in, d_raw, width, height);
    }

    {
        nvtx3::scoped_range r_thrust{"Thrust_Postprocess"};
        thrust::device_ptr<int16> t_raw(d_raw);
        thrust::device_ptr<unchar> t_fin(d_out_final);

        auto p = thrust::cuda::par.on(stream);

        thrust::for_each(p,
            thrust::make_zip_iterator(thrust::make_tuple(t_raw, t_fin)),
            thrust::make_zip_iterator(thrust::make_tuple(t_raw + size_pixels, t_fin + size_pixels)),
            Saturate());
    }
    cudaMemcpyAsync(output.data_ptr(), d_out_final, size_bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);

    cudaHostUnregister(input.data_ptr());
    cudaHostUnregister(output.data_ptr());
    cudaFree(d_in);
    cudaFree(d_raw);
    cudaFree(d_out_final);
    cudaStreamDestroy(stream);
    //cudaFreeAsync(d_raw, stream);

    return output;
}