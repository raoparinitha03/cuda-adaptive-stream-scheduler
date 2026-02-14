#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <vector>
#include "../common/task.h"

enum class Policy {
    EDF,
    SJF,
    LEAST_LOADED
};

class Scheduler {
public:
    Scheduler(Policy p);
    int select_gpu(const Task& task,
                   const std::vector<int>& gpu_load);

private:
    Policy policy;
};

#endif
