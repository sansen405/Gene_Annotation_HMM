#include "Viterbi.hpp"
#include "Intron_Duration.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace gene_hmm {

    using namespace std;

    vector<State> Viterbi::decode(
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model)
    {
        return decode(
            nucleotides,
            transition_log_probs,
            emission_model,
            0,
            numeric_limits<size_t>::max(),
            0.0);
    }

    vector<State> Viterbi::decode(
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        size_t min_intron_body_length,
        size_t max_intron_body_length)
    {
        return decode(
            nucleotides,
            transition_log_probs,
            emission_model,
            min_intron_body_length,
            max_intron_body_length,
            0.0);
    }

    vector<State> Viterbi::decode(
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        size_t min_intron_body_length,
        size_t max_intron_body_length,
        Log_Prob gene_start_penalty)
    {
        return decode(
            nucleotides,
            transition_log_probs,
            emission_model,
            min_intron_body_length,
            max_intron_body_length,
            gene_start_penalty,
            {});
    }

    vector<State> Viterbi::decode(
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        size_t min_intron_body_length,
        size_t max_intron_body_length,
        Log_Prob gene_start_penalty,
        const vector<Log_Prob>& intron_length_log_prob)
    {
        const size_t T = nucleotides.size();
        if (T == 0) return {};

        const bool use_length_model = !intron_length_log_prob.empty();

        // Streaming memory: keep only previous/current score + intron-length rows.
        // Backpointers remain O(T·S) but are 1-byte State values (~20× smaller than
        // storing full Log_Prob V and size_t length tables for every t).
        array<Log_Prob, NUM_STATES> prev_V{};
        array<Log_Prob, NUM_STATES> curr_V{};
        array<size_t, NUM_STATES> prev_intron_len{};
        array<size_t, NUM_STATES> curr_intron_len{};
        Backpointer_Matrix B(T, array<State, NUM_STATES>{});

        prev_V.fill(LOG_ZERO);
        curr_V.fill(LOG_ZERO);
        prev_intron_len.fill(0);
        curr_intron_len.fill(0);
        for(auto& row : B){
            row.fill(State::START);
        }

        for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
            prev_V[idx(s)]  = transition_log_probs[idx(State::START)][idx(s)];
            prev_V[idx(s)] += emission_model.emission_log_prob(s, 0, nucleotides);
            if(is_intron_body_state(s)){
                prev_intron_len[idx(s)] = 1;
            }
        }

        const auto& predecessors = emitting_predecessors();

        for(size_t t = 1; t < T; t++){
            curr_V.fill(LOG_ZERO);
            curr_intron_len.fill(0);

            for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
                Log_Prob max_prob = LOG_ZERO;
                State max_prob_prev = State::START;
                size_t max_prob_intron_body_length = 0;
                const Log_Prob emission = emission_model.emission_log_prob(s, t, nucleotides);

                for(State p : predecessors[idx(s)]){
                    if(!duration_allows_transition(
                        p,
                        s,
                        prev_intron_len[idx(p)],
                        min_intron_body_length,
                        max_intron_body_length)) {
                        continue;
                    }

                    Log_Prob transition_term = transition_log_probs[idx(p)][idx(s)];
                    if(use_length_model && is_intron_body_state(p) && p == s){
                        transition_term = 0.0;
                    }

                    Log_Prob curr_prob = prev_V[idx(p)] +
                                         transition_term -
                                         gene_entry_penalty(p, s, gene_start_penalty) +
                                         emission;

                    if(use_length_model && is_intron_body_state(p) && is_acceptor_state(s)){
                        curr_prob += length_log_prob_at(intron_length_log_prob, prev_intron_len[idx(p)]);
                    }

                    if(curr_prob > max_prob){
                        max_prob = curr_prob;
                        max_prob_prev = p;
                        if(is_intron_body_state(s)){
                            max_prob_intron_body_length =
                                (p == s) ? prev_intron_len[idx(p)] + 1 : 1;
                        } else {
                            max_prob_intron_body_length = 0;
                        }
                    }
                }
                curr_V[idx(s)] = max_prob;
                B[t][idx(s)] = max_prob_prev;
                curr_intron_len[idx(s)] = max_prob_intron_body_length;
            }

            prev_V.swap(curr_V);
            prev_intron_len.swap(curr_intron_len);
        }

        Log_Prob max_prob_final = LOG_ZERO;
        State final_state = State::START;
        for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
            Log_Prob curr_prob_final = prev_V[idx(s)] + transition_log_probs[idx(s)][idx(State::END)];
            if (curr_prob_final > max_prob_final){
                max_prob_final = curr_prob_final;
                final_state = s;
            }
        }

        vector<State> genome_annotation(T, State::START);
        genome_annotation[T - 1] = final_state;

        State curr_state = final_state;
        for (size_t t = T - 1; t > 0; t--){
            genome_annotation[t - 1] = B[t][idx(curr_state)];
            curr_state = genome_annotation[t - 1];
        }
        return genome_annotation;
    }

    Log_Prob Viterbi::path_log_prob(
        const vector<State>& states,
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        Log_Prob gene_start_penalty,
        size_t start,
        size_t end)
    {
        return path_log_prob(
            states,
            nucleotides,
            transition_log_probs,
            emission_model,
            gene_start_penalty,
            start,
            end,
            {});
    }

    Log_Prob Viterbi::path_log_prob(
        const vector<State>& states,
        const vector<Nucleotide>& nucleotides,
        const Transition_Model::Log_Prob_Matrix& transition_log_probs,
        const Emission_Model& emission_model,
        Log_Prob gene_start_penalty,
        size_t start,
        size_t end,
        const vector<Log_Prob>& intron_length_log_prob)
    {
        const bool use_length_model = !intron_length_log_prob.empty();
        Log_Prob total = 0.0;
        size_t intron_body_length = 0;

        // If scoring a mid-gene window, reconstruct intron dwell so far.
        if(start > 0 && start < states.size() && is_intron_body_state(states[start - 1])){
            size_t i = start - 1;
            while(i > 0 && states[i - 1] == states[i] && is_intron_body_state(states[i])){
                --i;
            }
            intron_body_length = start - i;
        }

        for(size_t t = start; t < end && t < states.size(); t++){
            total += emission_model.emission_log_prob(states[t], t, nucleotides);
            if(t == 0){
                total += transition_log_probs[idx(State::START)][idx(states[t])];
            } else {
                State previous = states[t - 1];
                State current = states[t];
                Log_Prob transition_term = transition_log_probs[idx(previous)][idx(current)];
                if(use_length_model && is_intron_body_state(previous) && previous == current){
                    transition_term = 0.0;
                }
                total += transition_term;
                total -= gene_entry_penalty(previous, current, gene_start_penalty);
                if(use_length_model && is_intron_body_state(previous) && is_acceptor_state(current)){
                    total += length_log_prob_at(intron_length_log_prob, intron_body_length);
                }
            }

            if(is_intron_body_state(states[t])){
                if(t > 0 && states[t - 1] == states[t]){
                    intron_body_length += 1;
                } else {
                    intron_body_length = 1;
                }
            } else {
                intron_body_length = 0;
            }
        }
        return total;
    }
}
