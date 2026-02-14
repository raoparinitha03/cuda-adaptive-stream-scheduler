#include "gpu_manager.h"
#include <iostream>
#include <limits>

extern void launch_kernel(float* d_data, int size, cudaStream_t stream);

GPUManager::GPUManager(int n, int s, int max_size)
    : num_gpus(n),
      streams_per_gpu(s),
      max_task_size(max_size),
      streams(n),
      device_buffers(n),
      avg_kernel_time_ms(n),
      pending_time_ms(n)
{
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);

        streams[g].resize(streams_per_gpu);
        device_buffers[g].resize(streams_per_gpu);
        avg_kernel_time_ms[g].resize(streams_per_gpu, 0.05f);
        pending_time_ms[g].resize(streams_per_gpu, 0.0f);

        for (int i = 0; i < streams_per_gpu; i++) {
            cudaStreamCreate(&streams[g][i]);
            cudaMalloc(&device_buffers[g][i],
                       max_task_size * sizeof(float));
        }
    }
}

GPUManager::~GPUManager() {
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);

        for (int i = 0; i < streams_per_gpu; i++) {
            cudaFree(device_buffers[g][i]);
            cudaStreamDestroy(streams[g][i]);
        }
    }
}

// Select stream with lowest predicted finish time
int GPUManager::select_stream(int gpu_id)
{
    float min_time = std::numeric_limits<float>::max();
    int best_stream = 0;

    for (int i = 0; i < streams_per_gpu; i++) {
        float predicted =
            pending_time_ms[gpu_id][i] +
            avg_kernel_time_ms[gpu_id][i];

        if (predicted < min_time) {
            min_time = predicted;
            best_stream = i;
        }
    }

    return best_stream;
}

void GPUManager::execute_task(int gpu_id, const Task& task)
{
    cudaSetDevice(0);

    int stream_id = select_stream(0);
    cudaStream_t stream = streams[0][stream_id];
    float* d_data = device_buffers[0][stream_id];

    pending_time_ms[0][stream_id] +=
        avg_kernel_time_ms[0][stream_id];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    launch_kernel(d_data, task.size, stream);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float latency_ms = 0.0f;
    cudaEventElapsedTime(&latency_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Update rolling average (exponential smoothing)
    float alpha = 0.2f;
    avg_kernel_time_ms[0][stream_id] =
        alpha * latency_ms +
        (1 - alpha) * avg_kernel_time_ms[0][stream_id];

    pending_time_ms[0][stream_id] -=
        avg_kernel_time_ms[0][stream_id];

    std::cout << "Task " << task.id
              << " | Stream: " << stream_id
              << " | Kernel Time: "
              << latency_ms << " ms"
              << " | Avg: "
              << avg_kernel_time_ms[0][stream_id]
              << " ms"
              << std::endl;
}

std::vector<float> GPUManager::get_stream_load(int gpu_id)
{
    return pending_time_ms[gpu_id];
}
