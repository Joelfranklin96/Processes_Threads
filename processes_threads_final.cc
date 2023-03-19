#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <omp.h>

int cpu_func(int result, int niters) {
    for (int i = 0; i < niters; ++i) {
        result = (result * result * i + 2 * result * i * i + 3) % 10000000;
    }
    return result;
}

class CpuThread {
public:
    CpuThread(int niters) : niters_(niters), result_(1) {}

    void operator()() {
        result_ = cpu_func(result_, niters_);
    }

    int getResult() const {
        return result_;
    }

private:
    int niters_;
    int result_;
};

class CpuProcess {
public:
    CpuProcess(int niters) : niters_(niters), result_(1) {}

    void operator()() {
        result_ = cpu_func(result_, niters_);
    }

    int getResult() const {
        return result_;
    }

private:
    int niters_;
    int result_;
};

class IoThread {
public:
    IoThread(int sleep) : sleep_(sleep), result_(sleep) {}

    void operator()() {
        std::this_thread::sleep_for(std::chrono::seconds(sleep_));
    }

    int getResult() const {
        return result_;
    }

private:
    int sleep_;
    int result_;
};

class IoProcess {
public:
    IoProcess(int sleep) : sleep_(sleep), result_(sleep) {}

    void operator()() {
        std::this_thread::sleep_for(std::chrono::seconds(sleep_));
    }

    int getResult() const {
        return result_;
    }

private:
    int sleep_;
    int result_;
};

template <typename ThreadType>
void run_threads(int nthreads, int work_size, std::vector<double>& results) {
    std::vector<ThreadType> thread_objs(nthreads, ThreadType(work_size));
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < nthreads; ++i) {
            thread_objs[i]();
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    results.push_back(duration.count());
}

template <typename ProcessType>
void run_processes(int nprocs, int work_size, std::vector<double>& results) {
    std::vector<ProcessType> process_objs(nprocs, ProcessType(work_size));
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(nprocs)
    {
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < nprocs; ++i) {
            process_objs[i]();
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    results.push_back(duration.count());
}

int main(int argc, char* argv[]) {
    int cpu_n_iters = std::stoi(argv[1]);
    int sleep = 1;
    int cpu_count = std::thread::hardware_concurrency();

    std::vector<std::pair<std::string, int>> input_params = {
        {"CpuThread", cpu_n_iters},
        {"CpuProcess", cpu_n_iters},
        {"IoThread", sleep},
        {"IoProcess", sleep}
    };

    std::vector<std::string> header = {"nthreads"};
    for (const auto& param : input_params) {
        header.push_back(param.first);
    }
    std::cout << std::setw(10) << std::left << header[0];
    for (size_t i = 1; i < header.size(); ++i) {
        std::cout << std::setw(15) << std::left << header[i];
    }
    std::cout << std::endl;

    for (int nthreads = 1; nthreads < 2 * cpu_count; ++nthreads) {
        std::vector<double> results;
        results.push_back(nthreads);
        for (const auto& param : input_params) {
            const std::string& name = param.first;
            int work_size = param.second;

            if (name == "CpuThread") {
                run_threads<CpuThread>(nthreads, work_size, results);
            } else if (name == "CpuProcess") {
                run_processes<CpuProcess>(nthreads, work_size, results);
            } else if (name == "IoThread") {
                run_threads<IoThread>(nthreads, work_size, results);
            } else if (name == "IoProcess") {
                run_processes<IoProcess>(nthreads, work_size, results);
            }
        }

        std::cout << std::setw(10) << std::left << results[0];
        for (size_t i = 1; i < results.size(); ++i) {
            std::cout << std::setw(15) << std::left << std::scientific << std::setprecision(6) << results[i];
        }
        std::cout << std::endl;
    }

    return 0;
}