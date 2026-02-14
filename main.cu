#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>

#define NUM_STREAMS 4
#define NUM_TASKS 40
#define DATA_SIZE (1 << 22)      // Increased workload
#define DEADLINE_MS 5.0f
__global__ void computeKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        #pragma unroll 50
        for (int i = 0; i < 50; i++) {
            val = val * 1.00001f + 0.00001f;
        }
        data[idx] = val;
    }
}
struct Task {
    int id;
    float deadline_ms;
};

struct PendingTask {
    cudaEvent_t start;
    cudaEvent_t stop;
    int task_id;
};

struct StreamState {
    cudaStream_t stream;

    float avg_exec_time_ms = 0.1f;
    float predicted_available_time = 0.0f;

    int completed_tasks = 0;
    float total_exec_time = 0.0f;

    std::vector<PendingTask> pending;
};
int main() {

    cudaFree(0);  // initialize CUDA context

    float* d_data;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(float));

    std::vector<float> h_data(DATA_SIZE, 1.0f);
    cudaMemcpy(d_data, h_data.data(),
               DATA_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    std::vector<StreamState> streams(NUM_STREAMS);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i].stream);
    }

    dim3 block(256);
    dim3 grid((DATA_SIZE + block.x - 1) / block.x);
    computeKernel<<<grid, block>>>(d_data, DATA_SIZE);
    cudaDeviceSynchronize();
    std::vector<Task> tasks;
    for (int i = 0; i < NUM_TASKS; i++) {
        tasks.push_back({i, DEADLINE_MS});
    }
    std::ofstream task_log("task_log.csv");
    task_log << "TaskID,Stream,ExecTime\n";

    int task_index = 0;

    auto global_start =
        std::chrono::high_resolution_clock::now();
    while (true) {

        // Launch new task if available
        if (task_index < NUM_TASKS) {

            // Simulate CPU preprocessing (overlap demo)
            std::this_thread::sleep_for(
                std::chrono::milliseconds(2));

            // Choose least-loaded stream
            int best_stream = 0;
            float min_finish =
                streams[0].predicted_available_time +
                streams[0].avg_exec_time_ms;

            for (int i = 1; i < NUM_STREAMS; i++) {
                float finish =
                    streams[i].predicted_available_time +
                    streams[i].avg_exec_time_ms;

                if (finish < min_finish) {
                    min_finish = finish;
                    best_stream = i;
                }
            }

            auto& chosen = streams[best_stream];

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, chosen.stream);

            computeKernel<<<grid, block, 0,
                            chosen.stream>>>(
                d_data, DATA_SIZE);

            cudaEventRecord(stop, chosen.stream);

            chosen.pending.push_back(
                {start, stop, task_index});

            chosen.predicted_available_time +=
                chosen.avg_exec_time_ms;

            std::cout << "Launched Task "
                      << task_index
                      << " on Stream "
                      << best_stream << std::endl;

            task_index++;
        }

        bool all_done =
            (task_index >= NUM_TASKS);
        for (int i = 0; i < NUM_STREAMS; i++) {

            auto& s = streams[i];

            for (auto it = s.pending.begin();
                 it != s.pending.end();) {

                if (cudaEventQuery(it->stop)
                    == cudaSuccess) {

                    float time_ms = 0.0f;
                    cudaEventElapsedTime(
                        &time_ms,
                        it->start,
                        it->stop);

                    s.completed_tasks++;
                    s.total_exec_time += time_ms;

                    s.avg_exec_time_ms =
                        s.total_exec_time /
                        s.completed_tasks;

                    // Log per-task data
                    task_log << it->task_id
                             << ","
                             << i
                             << ","
                             << time_ms
                             << "\n";

                    std::cout << "Completed Task "
                              << it->task_id
                              << " on Stream "
                              << i
                              << " | Time: "
                              << time_ms
                              << " ms\n";

                    cudaEventDestroy(it->start);
                    cudaEventDestroy(it->stop);

                    it = s.pending.erase(it);
                }
                else {
                    ++it;
                    all_done = false;
                }
            }
        }

        if (all_done) break;
    }

    cudaDeviceSynchronize();

    auto global_end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        total_time =
        global_end - global_start;

    std::cout << "\nTotal Wall Time: "
              << total_time.count()
              << " ms\n";

    task_log.close();
    std::ofstream file("stream_stats.csv");
    file << "Stream,CompletedTasks,TotalExecTime\n";

    for (int i = 0; i < NUM_STREAMS; i++) {
        file << i << ","
             << streams[i].completed_tasks
             << ","
             << streams[i].total_exec_time
             << "\n";
    }

    file.close();

    std::cout << "\n=== Final Stats ===\n";
    for (int i = 0; i < NUM_STREAMS; i++) {
        std::cout << "Stream " << i
                  << " | Tasks: "
                  << streams[i].completed_tasks
                  << " | Total Time: "
                  << streams[i].total_exec_time
                  << " ms\n";
    }

    cudaFree(d_data);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i].stream);
    }

    return 0;
}
