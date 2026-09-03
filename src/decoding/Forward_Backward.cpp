#include "Forward_Backward.hpp"
#include "Intron_Duration.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gene_hmm {

    using namespace std;

    static Log_Prob log_sum_exp(Log_Prob a, Log_Prob b) {
        if(a == LOG_ZERO) return b;
        if(b == LOG_ZERO) return a;
        Log_Prob max_value = max(a, b);
        return max_value + log(exp(a - max_value) + exp(b - max_value));
    }

    static size_t intron_frame_index(State state) {
        return idx(state) - idx(State::INTRON_1);
    }

    static Forward_Backward::Probability_Matrix make_probability_matrix(size_t T) {
        Forward_Backward::Probability_Matrix matrix(T, array<Log_Prob, NUM_STATES>{});
        for(auto& row : matrix){
            for(auto& value : row){
                value = LOG_ZERO;
            }
        }
        return matrix;
    }

    static Log_Prob sequence_log_prob(
        const Forward_Backward::Probability_Matrix& forward,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs)
    {
        Log_Prob total = LOG_ZERO;
        if(forward.empty()) return total;

        size_t last = forward.size() - 1;
        for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
            Log_Prob curr = forward[last][idx(s)] + transition_log_probs[idx(s)][idx(State::END)];
            total = log_sum_exp(total, curr);
        }
        return total;
    }

    Forward_Backward::Probability_Matrix Forward_Backward::posterior_log_probs(
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model)
    {
        return posterior_log_probs(
            nucleotides,
            transition_log_probs,
            emission_model,
            0.0,
            {},
            0);
    }

    Forward_Backward::Probability_Matrix Forward_Backward::posterior_log_probs(
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        Log_Prob gene_start_penalty)
    {
        return posterior_log_probs(
            nucleotides,
            transition_log_probs,
            emission_model,
            gene_start_penalty,
            {},
            0);
    }

    Forward_Backward::Probability_Matrix Forward_Backward::posterior_log_probs(
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        Log_Prob gene_start_penalty,
        const vector<Log_Prob>& intron_length_log_prob,
        size_t min_intron_body_length)
    {
        const size_t T = nucleotides.size();
        if(T == 0) return {};

        const bool use_length_model = !intron_length_log_prob.empty();
        const size_t max_duration = use_length_model
            ? max<size_t>(1, intron_length_log_prob.size() - 1)
            : 1;

        Probability_Matrix forward = make_probability_matrix(T);
        Probability_Matrix backward = make_probability_matrix(T);
        Probability_Matrix posterior = make_probability_matrix(T);

        // Duration-indexed mass for intron bodies (rolling over time).
        // Hard max: mass cannot extend past max_duration (matches Viterbi).
        // Hard min: acceptor exit only for d >= min_intron_body_length.
        using DurationRow = vector<array<Log_Prob, 3>>;
        DurationRow prev_intron(max_duration + 1);
        DurationRow curr_intron(max_duration + 1);
        for(auto& row : prev_intron) row = {LOG_ZERO, LOG_ZERO, LOG_ZERO};
        for(auto& row : curr_intron) row = {LOG_ZERO, LOG_ZERO, LOG_ZERO};

        const auto& predecessors = emitting_predecessors();
        const auto& successors = emitting_successors();

        for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
            forward[0][idx(s)] = transition_log_probs[idx(State::START)][idx(s)] +
                                 emission_model.emission_log_prob(s, 0, nucleotides);
            if(use_length_model && is_intron_body_state(s)){
                prev_intron[1][intron_frame_index(s)] = forward[0][idx(s)];
            }
        }

        for(size_t t = 1; t < T; t++){
            if(use_length_model){
                for(auto& row : curr_intron) row = {LOG_ZERO, LOG_ZERO, LOG_ZERO};
            }

            for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
                Log_Prob total = LOG_ZERO;
                const Log_Prob emission = emission_model.emission_log_prob(s, t, nucleotides);

                if(use_length_model && is_intron_body_state(s)){
                    const size_t frame = intron_frame_index(s);
                    // Enter from donor -> duration 1 (self-loop transition zeroed in length model).
                    for(State p : predecessors[idx(s)]){
                        if(is_donor_state(p)){
                            Log_Prob curr = forward[t - 1][idx(p)] +
                                            transition_log_probs[idx(p)][idx(s)] -
                                            gene_entry_penalty(p, s, gene_start_penalty) +
                                            emission;
                            curr_intron[1][frame] = log_sum_exp(curr_intron[1][frame], curr);
                        }
                    }
                    // Extend previous durations while d < max (hard max; no soft stay-at-cap).
                    for(size_t d = 1; d < max_duration; ++d){
                        if(prev_intron[d][frame] == LOG_ZERO) continue;
                        Log_Prob curr = prev_intron[d][frame] + emission;
                        curr_intron[d + 1][frame] = log_sum_exp(curr_intron[d + 1][frame], curr);
                    }
                    Log_Prob marginal = LOG_ZERO;
                    for(size_t d = 1; d <= max_duration; ++d){
                        marginal = log_sum_exp(marginal, curr_intron[d][frame]);
                    }
                    forward[t][idx(s)] = marginal;
                    continue;
                }

                for(State p : predecessors[idx(s)]){
                    if(use_length_model && is_intron_body_state(p) && is_acceptor_state(s)){
                        const size_t frame = intron_frame_index(p);
                        for(size_t d = 1; d <= max_duration; ++d){
                            if(d < min_intron_body_length) continue;
                            if(prev_intron[d][frame] == LOG_ZERO) continue;
                            Log_Prob curr = prev_intron[d][frame] +
                                            transition_log_probs[idx(p)][idx(s)] -
                                            gene_entry_penalty(p, s, gene_start_penalty) +
                                            length_log_prob_at(intron_length_log_prob, d) +
                                            emission;
                            total = log_sum_exp(total, curr);
                        }
                        continue;
                    }

                    Log_Prob transition_term = transition_log_probs[idx(p)][idx(s)];
                    if(use_length_model && is_intron_body_state(p) && p == s){
                        transition_term = 0.0;
                    }

                    Log_Prob curr = forward[t - 1][idx(p)] +
                                    transition_term -
                                    gene_entry_penalty(p, s, gene_start_penalty);
                    total = log_sum_exp(total, curr);
                }
                forward[t][idx(s)] = total + emission;
            }

            if(use_length_model){
                prev_intron.swap(curr_intron);
            }
        }

        for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
            backward[T - 1][idx(s)] = transition_log_probs[idx(s)][idx(State::END)];
        }

        // Backward with duration: beta_intron[frame][d] at time t.
        DurationRow next_intron_beta(max_duration + 1);
        DurationRow curr_intron_beta(max_duration + 1);
        for(auto& row : next_intron_beta) row = {LOG_ZERO, LOG_ZERO, LOG_ZERO};
        for(auto& row : curr_intron_beta) row = {LOG_ZERO, LOG_ZERO, LOG_ZERO};

        if(use_length_model){
            for(State s = State::INTRON_1; s <= State::INTRON_3; s = st(idx(s)+1)){
                const size_t frame = intron_frame_index(s);
                for(size_t d = 1; d <= max_duration; ++d){
                    next_intron_beta[d][frame] = backward[T - 1][idx(s)];
                }
            }
        }

        for(size_t t = T - 1; t > 0; t--){
            if(use_length_model){
                for(auto& row : curr_intron_beta) row = {LOG_ZERO, LOG_ZERO, LOG_ZERO};
            }

            for(State p = State::INTERGENIC; p < State::END; p = st(idx(p)+1)){
                Log_Prob total = LOG_ZERO;

                if(use_length_model && is_intron_body_state(p)){
                    const size_t frame = intron_frame_index(p);
                    for(size_t d = 1; d <= max_duration; ++d){
                        Log_Prob leave = LOG_ZERO;
                        if(d >= min_intron_body_length){
                            for(State s : successors[idx(p)]){
                                if(is_acceptor_state(s)){
                                    Log_Prob curr = transition_log_probs[idx(p)][idx(s)] -
                                                    gene_entry_penalty(p, s, gene_start_penalty) +
                                                    length_log_prob_at(intron_length_log_prob, d) +
                                                    emission_model.emission_log_prob(s, t, nucleotides) +
                                                    backward[t][idx(s)];
                                    leave = log_sum_exp(leave, curr);
                                }
                            }
                        }
                        // Hard max: stay only when d < max_duration (no soft stay-at-cap).
                        Log_Prob stay = LOG_ZERO;
                        if(d < max_duration){
                            stay = emission_model.emission_log_prob(p, t, nucleotides) +
                                   next_intron_beta[d + 1][frame];
                        }
                        curr_intron_beta[d][frame] = log_sum_exp(leave, stay);
                    }
                    Log_Prob marginal = LOG_ZERO;
                    for(size_t d = 1; d <= max_duration; ++d){
                        marginal = log_sum_exp(marginal, curr_intron_beta[d][frame]);
                    }
                    backward[t - 1][idx(p)] = marginal;
                    continue;
                }

                for(State s : successors[idx(p)]){
                    Log_Prob curr = transition_log_probs[idx(p)][idx(s)] -
                                    gene_entry_penalty(p, s, gene_start_penalty) +
                                    emission_model.emission_log_prob(s, t, nucleotides) +
                                    backward[t][idx(s)];
                    total = log_sum_exp(total, curr);
                }
                backward[t - 1][idx(p)] = total;
            }

            if(use_length_model){
                next_intron_beta.swap(curr_intron_beta);
            }
        }

        Log_Prob log_likelihood = sequence_log_prob(forward, transition_log_probs);
        for(size_t t = 0; t < T; t++){
            for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
                posterior[t][idx(s)] = forward[t][idx(s)] + backward[t][idx(s)] - log_likelihood;
            }
        }

        return posterior;
    }

    vector<double> Forward_Backward::confidence(
        const vector<Nucleotide>& nucleotides,
        const vector<State>& predicted_states,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model)
    {
        return confidence(
            nucleotides,
            predicted_states,
            transition_log_probs,
            emission_model,
            0.0,
            {},
            0);
    }

    vector<double> Forward_Backward::confidence(
        const vector<Nucleotide>& nucleotides,
        const vector<State>& predicted_states,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        Log_Prob gene_start_penalty)
    {
        return confidence(
            nucleotides,
            predicted_states,
            transition_log_probs,
            emission_model,
            gene_start_penalty,
            {},
            0);
    }

    vector<double> Forward_Backward::confidence(
        const vector<Nucleotide>& nucleotides,
        const vector<State>& predicted_states,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        Log_Prob gene_start_penalty,
        const vector<Log_Prob>& intron_length_log_prob,
        size_t min_intron_body_length)
    {
        if(nucleotides.size() != predicted_states.size()){
            throw runtime_error("Predicted state length does not match nucleotide length.");
        }

        Probability_Matrix posterior = posterior_log_probs(
            nucleotides,
            transition_log_probs,
            emission_model,
            gene_start_penalty,
            intron_length_log_prob,
            min_intron_body_length);

        vector<double> confidences(nucleotides.size(), 0.0);
        for(size_t t = 0; t < nucleotides.size(); t++){
            double value = exp(posterior[t][idx(predicted_states[t])]);
            confidences[t] = min(1.0, max(0.0, value));
        }
        return confidences;
    }
}
