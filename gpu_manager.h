#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <vector>
#include <cuda_runtime.h>
#include "../common/task.h"

class GPUManager {
public:
    GPUManager(int num_gpus, int streams_per_gpu, int max_task_size);
    ~GPUManager();

    int select_stream(int gpu_id);
    void execute_task(int gpu_id, const Task& task);
    std::vector<float> get_stream_load(int gpu_id);

private:
    int num_gpus;
    int streams_per_gpu;
    int max_task_size;

    std::vector<std::vector<cudaStream_t>> streams;
    std::vector<std::vector<float*>> device_buffers;

    // Adaptive scheduling data
    std::vector<std::vector<float>> avg_kernel_time_ms;
    std::vector<std::vector<float>> pending_time_ms;
};

#endif
