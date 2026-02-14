#ifndef TASK_H
#define TASK_H

#include <chrono>

struct Task {
    int id;
    int size;
    int deadline_ms;
    int estimated_time_ms;
    std::chrono::high_resolution_clock::time_point submit_time;
};

#endif
