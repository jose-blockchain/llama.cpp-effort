#include "server-cpu-usage.h"

#if !defined(_WIN32)

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <mach/mach_host.h>
#elif defined(__linux__)
#include <cstdio>
#include <fstream>
#endif

float get_cpu_usage_pct(uint64_t & prev_total, uint64_t & prev_idle, bool & prev_valid) {
#if defined(__APPLE__)
    /* macOS: all cores via host_processor_info (Mach) */
    natural_t num_cpus = 0;
    processor_cpu_load_info_data_t * cpu_info = nullptr;
    mach_msg_type_number_t count = 0;
    kern_return_t kr = host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO,
        &num_cpus, reinterpret_cast<processor_info_array_t *>(&cpu_info), &count);
    if (kr != KERN_SUCCESS || cpu_info == nullptr || num_cpus == 0) {
        return -1.0f;
    }
    uint64_t total = 0, idle = 0;
    for (natural_t i = 0; i < num_cpus; ++i) {
        total += cpu_info[i].cpu_ticks[CPU_STATE_USER] + cpu_info[i].cpu_ticks[CPU_STATE_SYSTEM]
            + cpu_info[i].cpu_ticks[CPU_STATE_NICE] + cpu_info[i].cpu_ticks[CPU_STATE_IDLE];
        idle += cpu_info[i].cpu_ticks[CPU_STATE_IDLE];
    }
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(cpu_info),
        count * sizeof(natural_t));
    if (!prev_valid) {
        prev_total = total;
        prev_idle = idle;
        prev_valid = true;
        return -1.0f;
    }
    uint64_t dt = total - prev_total;
    uint64_t di = idle - prev_idle;
    prev_total = total;
    prev_idle = idle;
    if (dt == 0) { return 0.0f; }
    float pct = 100.0f * (1.0f - (float)di / (float)dt);
    if (pct < 0.0f) { pct = 0.0f; }
    if (pct > 100.0f) { pct = 100.0f; }
    return pct;

#elif defined(__linux__)
    /* Linux: all cores from first line of /proc/stat (aggregate "cpu ") */
    std::ifstream f("/proc/stat");
    if (!f.is_open()) { return -1.0f; }
    char line[256];
    if (!f.getline(line, sizeof(line)) || line[0] != 'c' || line[1] != 'p' || line[2] != 'u' || line[3] != ' ') {
        return -1.0f;
    }
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
    int n = std::sscanf(line + 4, "%llu %llu %llu %llu %llu %llu %llu %llu",
        &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
    if (n < 4) { return -1.0f; }
    uint64_t total = user + nice + system + idle + (n > 4 ? iowait : 0) + (n > 5 ? irq : 0) + (n > 6 ? softirq : 0) + (n > 7 ? steal : 0);
    uint64_t id = idle + (n > 4 ? iowait : 0);
    if (!prev_valid) {
        prev_total = total;
        prev_idle = id;
        prev_valid = true;
        return -1.0f;
    }
    uint64_t dt = total - prev_total;
    uint64_t di = id - prev_idle;
    prev_total = total;
    prev_idle = id;
    if (dt == 0) { return 0.0f; }
    float pct = 100.0f * (1.0f - (float)di / (float)dt);
    if (pct < 0.0f) { pct = 0.0f; }
    if (pct > 100.0f) { pct = 100.0f; }
    return pct;

#else
    (void)prev_total;
    (void)prev_idle;
    (void)prev_valid;
    return -1.0f;
#endif
}

#endif
