#pragma once

#include <cstdint>

// Returns CPU usage as percentage 0-100 (all cores, system-wide).
// Implemented for macOS (Mach host_processor_info) and Linux (/proc/stat aggregate).
// Returns -1 if unavailable or on first sample. Pass persistent prev_total, prev_idle, prev_valid.
#if !defined(_WIN32)
float get_cpu_usage_pct(uint64_t & prev_total, uint64_t & prev_idle, bool & prev_valid);
#endif
