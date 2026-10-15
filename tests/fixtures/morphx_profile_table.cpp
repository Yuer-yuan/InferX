// Stress insertion and updates through the table used by the MorphX agent.
#include "morphx_profile_table.hpp"
#include <array>
#include <thread>
#include <vector>
int main() {
    MorphxProfileTable table;
    std::array<char, 4096> keys{};
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 8; ++worker) {
        threads.emplace_back([&] {
            for (int round = 0; round < 4; ++round)
                for (auto& key : keys) table[&key].fetch_add(1.f);
        });
    }
    for (auto& worker : threads) worker.join();
    for (auto& key : keys)
        if (table.count(&key) != 1 || table[&key].load() != 32.f) return 1;
    return 0;
}
