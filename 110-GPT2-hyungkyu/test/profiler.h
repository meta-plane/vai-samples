#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace vk {

struct ProfileEntry {
    std::string name;
    double total_ms = 0.0;
    int count = 0;

    double avg_ms() const { return count > 0 ? total_ms / count : 0.0; }
};

class Profiler {
public:
    static Profiler& instance() {
        static Profiler prof;
        return prof;
    }

    void begin(const std::string& name) {
        auto& entry = entries[name];
        entry.name = name;
        start_times[name] = std::chrono::high_resolution_clock::now();
    }

    void end(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto it = start_times.find(name);
        if (it != start_times.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - it->second).count() / 1000.0;

            auto& entry = entries[name];
            entry.total_ms += duration;
            entry.count++;

            start_times.erase(it);
        }
    }

    void reset() {
        entries.clear();
        start_times.clear();
    }

    void print(const std::string& title = "Profile Results") {
        if (entries.empty()) {
            std::cout << "No profiling data." << std::endl;
            return;
        }

        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  " << std::left << std::setw(54) << title << "  ║" << std::endl;
        std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ " << std::left << std::setw(35) << "Operation"
                  << std::right << std::setw(10) << "Total(ms)"
                  << std::setw(10) << "Avg(ms)"
                  << std::setw(8) << "Count" << " ║" << std::endl;
        std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;

        // Sort by total time (descending)
        std::vector<ProfileEntry> sorted;
        for (const auto& [name, entry] : entries) {
            sorted.push_back(entry);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const ProfileEntry& a, const ProfileEntry& b) {
                      return a.total_ms > b.total_ms;
                  });

        double total_time = 0.0;
        for (const auto& entry : sorted) {
            total_time += entry.total_ms;
        }

        for (const auto& entry : sorted) {
            double percent = total_time > 0 ? (entry.total_ms / total_time * 100.0) : 0.0;

            std::cout << "║ " << std::left << std::setw(35) << entry.name
                      << std::right << std::setw(10) << std::fixed << std::setprecision(2) << entry.total_ms
                      << std::setw(10) << entry.avg_ms()
                      << std::setw(8) << entry.count
                      << " ║ " << std::setprecision(1) << percent << "%" << std::endl;
        }

        std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ " << std::left << std::setw(35) << "TOTAL"
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << total_time
                  << std::setw(10) << " "
                  << std::setw(8) << " " << " ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    }

private:
    std::unordered_map<std::string, ProfileEntry> entries;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
};

// RAII helper for automatic timing
class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name(name) {
        Profiler::instance().begin(name);
    }
    ~ScopedTimer() {
        Profiler::instance().end(name);
    }
private:
    std::string name;
};

} // namespace vk

// Macro for easy profiling
#define PROFILE_SCOPE(name) vk::ScopedTimer _profile_timer_##__LINE__(name)
#define PROFILE_BEGIN(name) vk::Profiler::instance().begin(name)
#define PROFILE_END(name) vk::Profiler::instance().end(name)
#define PROFILE_PRINT() vk::Profiler::instance().print()
#define PROFILE_RESET() vk::Profiler::instance().reset()
