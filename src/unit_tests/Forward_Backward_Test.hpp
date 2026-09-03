#pragma once

#include "../decoding/Forward_Backward.hpp"
#include "Test_Utils.hpp"
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace gene_hmm {

    using namespace std;

    static Transition_Model::Log_Prob_Matrix make_forward_backward_log_zero_matrix() {
        Transition_Model::Log_Prob_Matrix matrix = {};
        for(auto& row : matrix){
            for(auto& value : row){
                value = LOG_ZERO;
            }
        }
        return matrix;
    }

    static void test_forward_backward_empty_sequence() {
        cout << "\n[TEST 1] Empty sequence returns empty posterior matrix\n";

        Emission_Model model;
        auto transitions = make_forward_backward_log_zero_matrix();
        auto posterior = Forward_Backward::posterior_log_probs({}, transitions, model);

        CHECK("empty input has no posterior rows", posterior.empty());
    }

    static void test_forward_backward_forced_path_confidence() {
        cout << "\n[TEST 2] Forced intergenic path has confidence 1 at each base\n";

        Emission_Model model;
        //donor emissions require loaded splice scores; an empty file gives neutral 0.0
        const string splice_scores_path = "/tmp/gene_hmm_fb_test_splice_scores.tsv";
        { ofstream empty_scores(splice_scores_path); }
        model.load_splice_cnn_scores(splice_scores_path, 3);

        vector<Nucleotide> nucs = {Nucleotide::C, Nucleotide::G, Nucleotide::T};
        vector<State> path = {State::INTERGENIC, State::INTERGENIC, State::INTERGENIC};
        auto transitions = make_forward_backward_log_zero_matrix();

        transitions[idx(State::START)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::END)] = 0.0;

        vector<double> confidence = Forward_Backward::confidence(nucs, path, transitions, model);

        CHECK("confidence length matches sequence length", confidence.size() == nucs.size());
        CHECK("all forced states have posterior confidence near 1",
              fabs(confidence[0] - 1.0) < 1e-9 &&
              fabs(confidence[1] - 1.0) < 1e-9 &&
              fabs(confidence[2] - 1.0) < 1e-9);
    }

    static void test_forward_backward_ambiguous_one_base_confidence() {
        cout << "\n[TEST 3] One-base posterior confidence follows transition mass\n";

        Emission_Model model;
        vector<Nucleotide> nucs = {Nucleotide::C};
        vector<State> path = {State::INTRON_1};
        auto transitions = make_forward_backward_log_zero_matrix();

        transitions[idx(State::START)][idx(State::INTERGENIC)] = log(0.25);
        transitions[idx(State::START)][idx(State::INTRON_1)] = log(0.75);
        transitions[idx(State::INTERGENIC)][idx(State::END)] = 0.0;
        transitions[idx(State::INTRON_1)][idx(State::END)] = 0.0;

        vector<double> confidence = Forward_Backward::confidence(nucs, path, transitions, model);

        CHECK("predicted intron state has posterior confidence near 0.75",
              confidence.size() == 1 && fabs(confidence[0] - 0.75) < 1e-9);
    }

    static void test_forward_backward_duration_model_accepts_length_term() {
        cout << "\n[TEST 4] Semi-Markov length model changes intron-exit posterior\n";

        Emission_Model model;
        const string splice_scores_path = "/tmp/gene_hmm_fb_duration_splice.tsv";
        const string start_scores_path = "/tmp/gene_hmm_fb_duration_start.tsv";
        { ofstream splice_file(splice_scores_path); }
        { ofstream start_file(start_scores_path); }

        // GT....AG so donor/acceptor motifs are legal; forced path uses only these states.
        // Positions: 0:G(donor1) 1:T(intron start) 2:A 3:A 4:A(acceptor AG at 3-4)
        vector<Nucleotide> nucs = {
            Nucleotide::G, Nucleotide::T, Nucleotide::A, Nucleotide::A, Nucleotide::G
        };
        model.load_splice_cnn_scores(splice_scores_path, nucs.size());
        model.load_start_cnn_scores(start_scores_path, nucs.size());

        auto transitions = make_forward_backward_log_zero_matrix();
        transitions[idx(State::START)][idx(State::DONOR_1)] = 0.0;
        transitions[idx(State::DONOR_1)][idx(State::INTRON_1)] = 0.0;
        transitions[idx(State::INTRON_1)][idx(State::INTRON_1)] = log(0.7);
        transitions[idx(State::INTRON_1)][idx(State::ACCEPTOR_1)] = log(0.3);
        transitions[idx(State::ACCEPTOR_1)][idx(State::END)] = 0.0;

        vector<Log_Prob> length_model(6, LOG_ZERO);
        length_model[1] = log(0.02);
        length_model[2] = log(0.02);
        length_model[3] = log(0.96);  // prefers duration 3 intron body bases

        auto posterior_plain = Forward_Backward::posterior_log_probs(
            nucs, transitions, model, 0.0, {});
        auto posterior_dur = Forward_Backward::posterior_log_probs(
            nucs, transitions, model, 0.0, length_model);

        CHECK("plain posterior has rows for each base", posterior_plain.size() == nucs.size());
        CHECK("duration posterior has rows for each base", posterior_dur.size() == nucs.size());

        bool changed = false;
        for(size_t t = 0; t < nucs.size(); ++t){
            for(State s = State::INTERGENIC; s < State::END; s = st(idx(s)+1)){
                if(fabs(posterior_plain[t][idx(s)] - posterior_dur[t][idx(s)]) > 1e-6){
                    changed = true;
                    break;
                }
            }
            if(changed) break;
        }
        CHECK("duration model changes some state posterior vs geometric", changed);
    }

    static void test_forward_backward_min_duration_gates_acceptor() {
        cout << "\n[TEST 5] Min intron duration hard-gates acceptor posterior mass\n";

        Emission_Model model;
        const string splice_scores_path = "/tmp/gene_hmm_fb_min_dur_splice.tsv";
        const string start_scores_path = "/tmp/gene_hmm_fb_min_dur_start.tsv";
        { ofstream splice_file(splice_scores_path); }
        { ofstream start_file(start_scores_path); }

        vector<Nucleotide> nucs = {
            Nucleotide::G, Nucleotide::T, Nucleotide::A, Nucleotide::A, Nucleotide::G
        };
        model.load_splice_cnn_scores(splice_scores_path, nucs.size());
        model.load_start_cnn_scores(start_scores_path, nucs.size());

        auto transitions = make_forward_backward_log_zero_matrix();
        transitions[idx(State::START)][idx(State::DONOR_1)] = log(0.5);
        transitions[idx(State::START)][idx(State::INTERGENIC)] = log(0.5);
        transitions[idx(State::DONOR_1)][idx(State::INTRON_1)] = 0.0;
        transitions[idx(State::INTRON_1)][idx(State::INTRON_1)] = log(0.5);
        transitions[idx(State::INTRON_1)][idx(State::ACCEPTOR_1)] = log(0.5);
        transitions[idx(State::ACCEPTOR_1)][idx(State::END)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::END)] = 0.0;

        vector<Log_Prob> length_model(6, log(1.0 / 5.0));
        length_model[0] = LOG_ZERO;

        auto posterior_open = Forward_Backward::posterior_log_probs(
            nucs, transitions, model, 0.0, length_model, 1);
        auto posterior_gated = Forward_Backward::posterior_log_probs(
            nucs, transitions, model, 0.0, length_model, 4);

        double open_acc = exp(posterior_open[4][idx(State::ACCEPTOR_1)]);
        double gated_acc = exp(posterior_gated[4][idx(State::ACCEPTOR_1)]);
        CHECK("open min allows some acceptor posterior mass", open_acc > 1e-9);
        CHECK("min=4 blocks short-intron acceptor mass at AG",
              gated_acc < open_acc * 0.5 + 1e-12);
    }

    static void run_Forward_Backward_tests() {
        cout << "\nRunning Forward-Backward tests...\n";

        test_forward_backward_empty_sequence();
        test_forward_backward_forced_path_confidence();
        test_forward_backward_ambiguous_one_base_confidence();
        test_forward_backward_duration_model_accepts_length_term();
        test_forward_backward_min_duration_gates_acceptor();

        cout << "\nDone.\n";
    }
}
