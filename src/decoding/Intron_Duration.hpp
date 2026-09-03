#pragma once

#include "../topology/Topology.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace gene_hmm {

    using namespace std;

    inline bool is_intron_body_state(State state) {
        return state == State::INTRON_1 ||
               state == State::INTRON_2 ||
               state == State::INTRON_3;
    }

    inline bool is_acceptor_state(State state) {
        return state == State::ACCEPTOR_1 ||
               state == State::ACCEPTOR_2 ||
               state == State::ACCEPTOR_3;
    }

    inline bool is_donor_state(State state) {
        return state == State::DONOR_1 ||
               state == State::DONOR_2 ||
               state == State::DONOR_3;
    }

    inline Log_Prob gene_entry_penalty(State previous, State current, Log_Prob penalty) {
        if(previous == State::INTERGENIC && current == State::START_CODON_1) {
            return penalty;
        }
        return 0.0;
    }

    // Hard gates shared by Viterbi and Forward-Backward:
    // self-loop only while length < max; acceptor exit only when length >= min.
    inline bool duration_allows_transition(
        State previous,
        State current,
        size_t previous_intron_body_length,
        size_t min_intron_body_length,
        size_t max_intron_body_length)
    {
        if(is_intron_body_state(previous) && previous == current){
            return previous_intron_body_length < max_intron_body_length;
        }
        if(is_intron_body_state(previous) && is_acceptor_state(current)){
            return previous_intron_body_length >= min_intron_body_length;
        }
        return true;
    }

    inline Log_Prob length_log_prob_at(
        const vector<Log_Prob>& intron_length_log_prob,
        size_t length)
    {
        if(intron_length_log_prob.empty()) return 0.0;
        size_t index = min(length, intron_length_log_prob.size() - 1);
        return intron_length_log_prob[index];
    }

    inline size_t percentile_length(vector<size_t> lengths, double percentile) {
        if(lengths.empty()) {
            return numeric_limits<size_t>::max();
        }
        sort(lengths.begin(), lengths.end());
        size_t index = static_cast<size_t>(percentile * static_cast<double>(lengths.size() - 1));
        return lengths[index];
    }

    // Laplace-smoothed empirical histogram over intron body lengths in [1, max_length].
    inline vector<Log_Prob> build_histogram_intron_length_log_probs(
        const vector<size_t>& lengths,
        size_t max_length)
    {
        if(max_length == 0 || lengths.empty()) {
            return {};
        }

        vector<double> counts(max_length + 1, 1.0);
        counts[0] = 0.0;
        for(size_t length : lengths) {
            if(length >= 1 && length <= max_length) {
                counts[length] += 1.0;
            }
        }

        double total = 0.0;
        for(size_t length = 1; length <= max_length; ++length) {
            total += counts[length];
        }

        vector<Log_Prob> log_probs(max_length + 1, LOG_ZERO);
        for(size_t length = 1; length <= max_length; ++length) {
            log_probs[length] = log(counts[length] / total);
        }
        return log_probs;
    }

    // Negative-binomial duration on support [1, max_length].
    // Parameterized by mean m and success probability p in (0,1), with
    // r = m * p / (1-p) so E[L] ≈ m under the usual NB(r, p) "number of failures" form shifted to start at 1.
    inline vector<Log_Prob> build_negative_binomial_intron_length_log_probs(
        const vector<size_t>& lengths,
        size_t max_length)
    {
        if(max_length == 0 || lengths.empty()) {
            return {};
        }

        double sum = 0.0;
        double sum_sq = 0.0;
        size_t n = 0;
        for(size_t length : lengths) {
            if(length < 1 || length > max_length) continue;
            sum += static_cast<double>(length);
            sum_sq += static_cast<double>(length) * static_cast<double>(length);
            ++n;
        }
        if(n == 0) {
            return {};
        }

        double mean = sum / static_cast<double>(n);
        double var = max(mean + 1e-6, sum_sq / static_cast<double>(n) - mean * mean);
        // NB: var = mean + mean^2 / r  =>  r = mean^2 / (var - mean)
        double r = (var > mean)
            ? (mean * mean) / (var - mean)
            : mean;  // near-Poisson fallback
        r = max(1e-3, r);
        double p = r / (r + mean);  // success probability
        p = min(1.0 - 1e-9, max(1e-9, p));

        auto log_gamma = [](double x) -> double {
            return lgamma(x);
        };

        vector<double> unnormalized(max_length + 1, 0.0);
        double total = 0.0;
        for(size_t length = 1; length <= max_length; ++length) {
            // P(K=k) for k failures before r successes: C(k+r-1, k) * (1-p)^k * p^r
            // with k = length (body length as failure count).
            double k = static_cast<double>(length);
            double log_pmf =
                log_gamma(k + r) - log_gamma(k + 1.0) - log_gamma(r) +
                k * log(1.0 - p) + r * log(p);
            unnormalized[length] = exp(log_pmf);
            total += unnormalized[length];
        }

        vector<Log_Prob> log_probs(max_length + 1, LOG_ZERO);
        if(total <= 0.0) {
            return {};
        }
        for(size_t length = 1; length <= max_length; ++length) {
            log_probs[length] = log(unnormalized[length] / total);
        }
        return log_probs;
    }

    enum class Intron_Duration_Kind {
        Histogram,
        NegativeBinomial,
        None
    };

    inline vector<Log_Prob> build_intron_length_log_probs(
        const vector<size_t>& lengths,
        size_t max_length,
        Intron_Duration_Kind kind = Intron_Duration_Kind::Histogram)
    {
        switch(kind) {
            case Intron_Duration_Kind::NegativeBinomial:
                return build_negative_binomial_intron_length_log_probs(lengths, max_length);
            case Intron_Duration_Kind::None:
                return {};
            case Intron_Duration_Kind::Histogram:
            default:
                return build_histogram_intron_length_log_probs(lengths, max_length);
        }
    }

    // Implied mean geometric dwell from intron self-loop probability a:
    // E[L] = 1 / (1 - a) for geometric starting at 1.
    inline double geometric_mean_dwell_from_self_loop(Log_Prob self_loop_log_prob) {
        if(self_loop_log_prob <= LOG_ZERO) return 1.0;
        double a = exp(self_loop_log_prob);
        if(a >= 1.0) return numeric_limits<double>::infinity();
        return 1.0 / (1.0 - a);
    }

    struct Duration_Dwell_Summary {
        double empirical_mean = 0.0;
        double empirical_p50 = 0.0;
        double empirical_p95 = 0.0;
        double geometric_mean_from_A = 0.0;
        double histogram_mean = 0.0;
        double negative_binomial_mean = 0.0;
        size_t n_lengths = 0;
        size_t max_length = 0;
    };

    inline Duration_Dwell_Summary summarize_duration_dwell(
        const vector<size_t>& lengths,
        size_t max_length,
        Log_Prob intron_self_loop_log_prob)
    {
        Duration_Dwell_Summary summary;
        summary.n_lengths = lengths.size();
        summary.max_length = max_length;
        summary.geometric_mean_from_A = geometric_mean_dwell_from_self_loop(intron_self_loop_log_prob);

        if(lengths.empty()) return summary;

        double sum = 0.0;
        for(size_t length : lengths) sum += static_cast<double>(length);
        summary.empirical_mean = sum / static_cast<double>(lengths.size());
        summary.empirical_p50 = static_cast<double>(percentile_length(lengths, 0.50));
        summary.empirical_p95 = static_cast<double>(percentile_length(lengths, 0.95));

        auto hist = build_histogram_intron_length_log_probs(lengths, max_length);
        auto nb = build_negative_binomial_intron_length_log_probs(lengths, max_length);
        double hist_mean = 0.0;
        double nb_mean = 0.0;
        for(size_t length = 1; length <= max_length && length < hist.size(); ++length) {
            if(hist[length] > LOG_ZERO) {
                hist_mean += static_cast<double>(length) * exp(hist[length]);
            }
            if(length < nb.size() && nb[length] > LOG_ZERO) {
                nb_mean += static_cast<double>(length) * exp(nb[length]);
            }
        }
        summary.histogram_mean = hist_mean;
        summary.negative_binomial_mean = nb_mean;
        return summary;
    }

}
