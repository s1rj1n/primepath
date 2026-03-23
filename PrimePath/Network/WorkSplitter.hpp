#pragma once
#include "../TaskManager.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace prime { namespace net {

// =========================================================================
// WorkChunk -- a discrete unit of work assigned to a Carriage
// =========================================================================

struct WorkChunk {
    std::string task_id;
    TaskType    type        = TaskType::GeneralPrime;
    uint64_t    range_start = 0;
    uint64_t    range_end   = 0;
};

// =========================================================================
// Generate a unique task ID from type + range
// =========================================================================

inline std::string make_task_id(TaskType type, uint64_t start, uint64_t end) {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

    std::ostringstream os;
    os << task_key(type) << "_"
       << std::hex << start << "_" << end << "_"
       << std::hex << (ms & 0xFFFFFF);
    return os.str();
}

// =========================================================================
// split_range -- divide a range into roughly equal chunks for N workers
//
// For most task types we split the numeric range evenly.
// Some task types (Wieferich, Wilson) are much heavier per candidate,
// so callers may want to use smaller ranges for those -- but that is a
// policy decision for the Conductor, not the splitter.
// =========================================================================

inline std::vector<WorkChunk> split_range(TaskType type,
                                           uint64_t start,
                                           uint64_t end,
                                           int num_workers) {
    std::vector<WorkChunk> chunks;
    if (num_workers <= 0 || end <= start) return chunks;

    uint64_t total = end - start;
    uint64_t base_size = total / static_cast<uint64_t>(num_workers);
    uint64_t remainder = total % static_cast<uint64_t>(num_workers);

    uint64_t pos = start;
    for (int i = 0; i < num_workers && pos < end; ++i) {
        uint64_t chunk_size = base_size + (static_cast<uint64_t>(i) < remainder ? 1 : 0);
        if (chunk_size == 0) break;

        uint64_t chunk_end = pos + chunk_size;
        if (chunk_end > end) chunk_end = end;

        WorkChunk wc;
        wc.task_id     = make_task_id(type, pos, chunk_end);
        wc.type        = type;
        wc.range_start = pos;
        wc.range_end   = chunk_end;
        chunks.push_back(std::move(wc));

        pos = chunk_end;
    }

    return chunks;
}

// =========================================================================
// reassign_failed -- create a new chunk from where a failed worker left off
//
// If a Carriage disconnects while processing a chunk, the Conductor calls
// this to create a replacement chunk covering the remaining range.
// =========================================================================

inline WorkChunk reassign_failed(const WorkChunk& failed, uint64_t last_known_pos) {
    WorkChunk wc;
    wc.type = failed.type;

    // Resume from last known position (or from the start if no progress)
    uint64_t resume_from = last_known_pos;
    if (resume_from < failed.range_start || resume_from >= failed.range_end) {
        resume_from = failed.range_start;
    }

    wc.range_start = resume_from;
    wc.range_end   = failed.range_end;
    wc.task_id     = make_task_id(wc.type, wc.range_start, wc.range_end);
    return wc;
}

}} // namespace prime::net
