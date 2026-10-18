#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <stdexcept>

namespace ninfer::runtime {
// Fixed before the executor worker starts. OFF preserves the original batch estimator.
inline bool completion_request_yield_max_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SCHED_REQUEST_YIELD_MAX");
        if (value == nullptr || std::string_view(value) == "0") { return false; }
        if (std::string_view(value) == "1") { return true; }
        throw std::invalid_argument("NINFER_SCHED_REQUEST_YIELD_MAX supports 0/1");
    }();
    return enabled;
}

struct CompletionBudgetView {
    double first_token_ms = 0;
    unsigned committed_tokens = 0;
    unsigned remaining_tokens = 0;
    std::uint64_t committed_decode_rounds = 0;
};

enum class CompletionChoice { Prefill, Decode, PrefillDeadlineConflict };
struct BatchCostBin {
    double round_ms = 0;
    double tokens_per_row_round = 3;
    unsigned observations = 0;
};

template <std::size_t Capacity>
class CompletionCostModel {
public:
    BatchCostBin get(unsigned rows) const {
        rows = std::clamp(rows, 1U, static_cast<unsigned>(Capacity));
        auto bin = bins_[rows - 1];
        if (!bin.observations) { bin.round_ms = 25.0 + 12.0 * (rows - 1); }
        return bin;
    }
    void observe(unsigned rows, double elapsed_ms, unsigned committed_tokens) {
        if (!rows || rows > Capacity || !committed_tokens || elapsed_ms <= 0 ||
            !std::isfinite(elapsed_ms)) { return; }
        auto& bin = bins_[rows - 1];
        const double yield = static_cast<double>(committed_tokens) / rows;
        const double alpha = bin.observations ? .2 : 1.0;
        bin.round_ms += alpha * (elapsed_ms - bin.round_ms);
        bin.tokens_per_row_round += alpha * (yield - bin.tokens_per_row_round);
        ++bin.observations;
    }
private:
    std::array<BatchCostBin, Capacity> bins_{};
};

// Request.generated includes one prefill token. Count only resolved, noncancelled
// decode rounds and actual committed tokens; fast rows never reduce the old estimate.
inline double completion_remaining_rounds(const CompletionBudgetView& request,
                                          double batch_tokens_per_round) {
    const double batch_rounds = std::ceil(
        request.remaining_tokens / std::max(1.0, batch_tokens_per_round));
    if (!request.committed_decode_rounds || request.committed_tokens <= 1) {
        return batch_rounds;
    }
    const double own_yield = static_cast<double>(request.committed_tokens - 1) /
                            static_cast<double>(request.committed_decode_rounds);
    const double own_rounds = std::ceil(request.remaining_tokens / std::max(1.0, own_yield));
    return std::max(batch_rounds, own_rounds);
}

inline CompletionChoice choose_completion_unit(
    double now_ms, double prefill_deadline_ms, double remaining_prefill_ms,
    double next_chunk_ms, BatchCostBin decode, double tpot_ms,
    std::span<const CompletionBudgetView> decodes) {
    if (decodes.empty()) { return CompletionChoice::Prefill; }
    double min_decode_laxity = std::numeric_limits<double>::infinity();
    for (const auto& d : decodes) {
        if (!d.remaining_tokens) { continue; }
        const unsigned total = d.committed_tokens + d.remaining_tokens;
        const double deadline = d.first_token_ms + (total - 1) * tpot_ms;
        const double remaining_rounds =
            completion_remaining_rounds(d, decode.tokens_per_row_round);
        min_decode_laxity = std::min(min_decode_laxity,
            deadline - now_ms - remaining_rounds * decode.round_ms);
    }
    const double prefill_laxity = prefill_deadline_ms - now_ms - remaining_prefill_ms;
    // Preserve FCFS progress even when both workload targets are infeasible.
    if (prefill_laxity <= decode.round_ms) {
        return min_decode_laxity < next_chunk_ms
            ? CompletionChoice::PrefillDeadlineConflict : CompletionChoice::Prefill;
    }
    // Earliest latest-start time over whole completion work, not a token-prefix
    // rate constraint. Look ahead one chunk because it is not preemptible.
    return min_decode_laxity - next_chunk_ms < prefill_laxity - decode.round_ms
        ? CompletionChoice::Decode : CompletionChoice::Prefill;
}
} // namespace ninfer::runtime
