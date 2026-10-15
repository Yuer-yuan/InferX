#pragma once
#include <atomic>
#include <mutex>
#include <unordered_map>

// References survive rehash; entries are never erased while the agent runs.
class MorphxProfileTable {
    std::unordered_map<const char*, std::atomic<float>> values;
    mutable std::mutex mutex;
public:
    std::atomic<float>& operator[](const char* key) {
        std::lock_guard lock(mutex);
        return values.try_emplace(key, 0.f).first->second;
    }
    size_t count(const char* key) const {
        std::lock_guard lock(mutex);
        return values.count(key);
    }
};
