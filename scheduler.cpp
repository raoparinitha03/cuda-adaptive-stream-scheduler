#include "scheduler.h"
#include <algorithm>

Scheduler::Scheduler(Policy p) : policy(p) {}

int Scheduler::select_gpu(const Task& task,
                          const std::vector<int>& gpu_load)
{
    return std::distance(gpu_load.begin(),
           std::min_element(gpu_load.begin(), gpu_load.end()));
}
