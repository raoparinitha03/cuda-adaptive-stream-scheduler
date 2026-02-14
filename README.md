# cuda-adaptive-stream-scheduler

## Overview

This project implements an adaptive GPU task scheduler using CUDA streams.
It dynamically distributes tasks across multiple streams using a least-finish-time heuristic to achieve balanced workload distribution and CPU–GPU overlap.

The system measures execution time using CUDA events and evaluates load fairness across streams.

## Features

Multi-stream asynchronous kernel execution

Event-based GPU timing

Adaptive least-finish-time scheduling

CPU–GPU overlap simulation

Per-task execution logging

Load imbalance ratio analysis

Visualization-ready CSV output.

## Architecture

Tasks are generated dynamically.

Each task is assigned to the stream predicted to finish earliest.

Kernels are launched asynchronously.

CUDA events are used for accurate GPU timing.

A runtime loop polls for task completion.

Execution statistics are logged for analysis.

## Scheduling Strategy

The scheduler selects the stream with minimum:

predicted_available_time + avg_execution_time

This approximates a least-finish-time heuristic commonly used in distributed systems scheduling.

## Results

Under uniform workloads:

Load Imbalance Ratio ≈ 1.009

This indicates near-optimal workload distribution across streams.

Build Instructions

## Requires:

NVIDIA GPU

CUDA Toolkit installed

## Compile:

nvcc src/main.cu -o scheduler
./scheduler


## Output Files

task_log.csv → Per-task execution time

stream_stats.csv → Per-stream summary statistics

These can be used for visualization and performance analysis.
