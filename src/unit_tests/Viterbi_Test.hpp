#pragma once

#include "../decoding/Viterbi.hpp"
#include "Test_Utils.hpp"
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace gene_hmm {

    using namespace std;

    static Transition_Model::Log_Prob_Matrix make_viterbi_log_zero_matrix() {
        Transition_Model::Log_Prob_Matrix matrix = {};
        for (auto& row : matrix) {
            for (auto& value : row) {
                value = LOG_ZERO;
            }
        }
        return matrix;
    }

    //donor/acceptor/start emissions require loaded CNN scores; empty files give neutral 0.0
    static Emission_Model make_viterbi_test_model(size_t length) {
        Emission_Model model;
        const string splice_path = "/tmp/gene_hmm_viterbi_test_splice_scores.tsv";
        const string start_path = "/tmp/gene_hmm_viterbi_test_start_scores.tsv";
        { ofstream splice_file(splice_path); }
        { ofstream start_file(start_path); }
        model.load_splice_cnn_scores(splice_path, length);
        model.load_start_cnn_scores(start_path, length);
        return model;
    }

    static void test_viterbi_empty_sequence() {
        cout << "\n[TEST 1] Empty sequence returns empty path\n";

        Emission_Model model;
        auto transitions = make_viterbi_log_zero_matrix();
        vector<State> path = Viterbi::decode({}, transitions, model);

        CHECK("empty input decodes to empty annotation", path.empty());
    }

    static void test_viterbi_deterministic_start_codon_path() {
        cout << "\n[TEST 2] Deterministic path decodes ATG start codon\n";

        vector<Nucleotide> nucs = {Nucleotide::A, Nucleotide::T, Nucleotide::G};
        Emission_Model model = make_viterbi_test_model(nucs.size());
        auto transitions = make_viterbi_log_zero_matrix();
        transitions[idx(State::START)][idx(State::START_CODON_1)] = 0.0;
        transitions[idx(State::START_CODON_1)][idx(State::START_CODON_2)] = 0.0;
        transitions[idx(State::START_CODON_2)][idx(State::START_CODON_3)] = 0.0;
        transitions[idx(State::START_CODON_3)][idx(State::END)] = 0.0;

        vector<State> path = Viterbi::decode(nucs, transitions, model);
        vector<State> expected = {
            State::START_CODON_1,
            State::START_CODON_2,
            State::START_CODON_3
        };

        CHECK("decoded path length matches sequence length", path.size() == nucs.size());
        CHECK("decoded path follows forced start-codon states", path == expected);
    }

    static void test_viterbi_intergenic_self_loop_path() {
        cout << "\n[TEST 3] Intergenic self-loop decodes repeated intergenic states\n";

        vector<Nucleotide> nucs = {Nucleotide::C, Nucleotide::G, Nucleotide::T, Nucleotide::A};
        Emission_Model model = make_viterbi_test_model(nucs.size());
        auto transitions = make_viterbi_log_zero_matrix();
        transitions[idx(State::START)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::END)] = 0.0;

        vector<State> path = Viterbi::decode(nucs, transitions, model);

        CHECK("decoded path length matches sequence length", path.size() == nucs.size());
        CHECK("all decoded states are INTERGENIC",
              path == vector<State>(nucs.size(), State::INTERGENIC));
    }

    static void test_viterbi_chooses_higher_scoring_path() {
        cout << "\n[TEST 4] Decoder chooses higher-scoring transition path\n";

        vector<Nucleotide> nucs = {Nucleotide::A, Nucleotide::T, Nucleotide::G};
        Emission_Model model = make_viterbi_test_model(nucs.size());
        auto transitions = make_viterbi_log_zero_matrix();

        transitions[idx(State::START)][idx(State::INTERGENIC)] = log(0.1);
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = log(0.1);

        transitions[idx(State::START)][idx(State::START_CODON_1)] = log(0.9);
        transitions[idx(State::START_CODON_1)][idx(State::START_CODON_2)] = log(0.9);
        transitions[idx(State::START_CODON_2)][idx(State::START_CODON_3)] = log(0.9);
        transitions[idx(State::START_CODON_3)][idx(State::END)] = log(0.9);

        vector<State> path = Viterbi::decode(nucs, transitions, model);

        CHECK("higher-scoring ATG path wins over intergenic alternative",
              path == vector<State>({
                  State::START_CODON_1,
                  State::START_CODON_2,
                  State::START_CODON_3
              }));
    }

    static void test_viterbi_intron_body_min_length() {
        cout << "\n[TEST 5] Intron body duration constraint gates acceptor exit\n";

        vector<Nucleotide> nucs = {Nucleotide::C, Nucleotide::A, Nucleotide::G};
        Emission_Model model = make_viterbi_test_model(nucs.size());
        auto transitions = make_viterbi_log_zero_matrix();

        transitions[idx(State::START)][idx(State::INTRON_1)] = 0.0;
        transitions[idx(State::INTRON_1)][idx(State::INTRON_1)] = 0.0;
        transitions[idx(State::INTRON_1)][idx(State::ACCEPTOR_1)] = 0.0;
        transitions[idx(State::ACCEPTOR_1)][idx(State::END)] = 0.0;

        transitions[idx(State::START)][idx(State::INTERGENIC)] = log(0.01);
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = log(0.01);
        transitions[idx(State::INTERGENIC)][idx(State::END)] = log(0.01);

        vector<State> allowed_path = Viterbi::decode(nucs, transitions, model, 2, 10);
        vector<State> blocked_path = Viterbi::decode(nucs, transitions, model, 3, 10);

        CHECK("acceptor exit is allowed when intron body meets minimum length",
              allowed_path == vector<State>({State::INTRON_1, State::INTRON_1, State::ACCEPTOR_1}));
        CHECK("acceptor exit is blocked when intron body is too short",
              blocked_path == vector<State>(nucs.size(), State::INTERGENIC));
    }

    static void test_viterbi_gene_start_penalty() {
        cout << "\n[TEST 6] Gene start penalty discourages intergenic gene entry\n";

        vector<Nucleotide> nucs = {Nucleotide::C, Nucleotide::A, Nucleotide::T, Nucleotide::G};
        Emission_Model model = make_viterbi_test_model(nucs.size());
        auto transitions = make_viterbi_log_zero_matrix();

        transitions[idx(State::START)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = log(0.8);
        transitions[idx(State::INTERGENIC)][idx(State::START_CODON_1)] = log(0.9);
        transitions[idx(State::START_CODON_1)][idx(State::START_CODON_2)] = log(0.9);
        transitions[idx(State::START_CODON_2)][idx(State::START_CODON_3)] = log(0.9);
        transitions[idx(State::START_CODON_3)][idx(State::END)] = log(0.9);
        transitions[idx(State::INTERGENIC)][idx(State::END)] = log(0.8);

        vector<State> unpenalized_path = Viterbi::decode(
            nucs,
            transitions,
            model,
            0,
            numeric_limits<size_t>::max(),
            0.0);
        vector<State> penalized_path = Viterbi::decode(
            nucs,
            transitions,
            model,
            0,
            numeric_limits<size_t>::max(),
            5.0);

        CHECK("unpenalized path enters a gene",
              unpenalized_path == vector<State>({
                  State::INTERGENIC,
                  State::START_CODON_1,
                  State::START_CODON_2,
                  State::START_CODON_3
              }));
        CHECK("high gene start penalty keeps path intergenic",
              penalized_path == vector<State>(nucs.size(), State::INTERGENIC));
    }

    static void test_viterbi_path_log_prob_matches_duration_decode_score() {
        cout << "\n[TEST 7] path_log_prob HSMM terms match explicit duration accounting\n";

        Emission_Model model = make_viterbi_test_model(3);
        auto transitions = make_viterbi_log_zero_matrix();
        transitions[idx(State::START)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = log(0.7);
        transitions[idx(State::INTERGENIC)][idx(State::END)] = log(0.3);
        vector<Nucleotide> nucs = {Nucleotide::C, Nucleotide::G, Nucleotide::T};
        vector<State> path = {State::INTERGENIC, State::INTERGENIC, State::INTERGENIC};

        Log_Prob plain = Viterbi::path_log_prob(
            path, nucs, transitions, model, 0.0, 0, path.size(), {});
        Log_Prob with_empty_table = Viterbi::path_log_prob(
            path, nucs, transitions, model, 0.0, 0, path.size());
        CHECK("overload without duration matches empty duration table",
              fabs(plain - with_empty_table) < 1e-12);

        // Duration path: donor + 3 intron body bases + acceptor.
        nucs = {
            Nucleotide::G, Nucleotide::T, Nucleotide::A, Nucleotide::A, Nucleotide::G
        };
        model = make_viterbi_test_model(nucs.size());
        transitions = make_viterbi_log_zero_matrix();
        transitions[idx(State::START)][idx(State::DONOR_1)] = 0.0;
        transitions[idx(State::DONOR_1)][idx(State::INTRON_1)] = log(0.5);
        transitions[idx(State::INTRON_1)][idx(State::INTRON_1)] = log(0.4);
        transitions[idx(State::INTRON_1)][idx(State::ACCEPTOR_1)] = log(0.6);
        transitions[idx(State::ACCEPTOR_1)][idx(State::END)] = 0.0;

        path = {
            State::DONOR_1,
            State::INTRON_1,
            State::INTRON_1,
            State::INTRON_1,
            State::ACCEPTOR_1
        };

        vector<Log_Prob> length_model(5, LOG_ZERO);
        length_model[3] = log(1.0);  // only duration 3 has mass

        Log_Prob geometric = Viterbi::path_log_prob(
            path, nucs, transitions, model, 0.0, 0, path.size(), {});
        Log_Prob hsmm = Viterbi::path_log_prob(
            path, nucs, transitions, model, 0.0, 0, path.size(), length_model);

        // Geometric includes two intron self-loops of log(0.4); HSMM zeroes them and
        // charges log P(L=3)=0 at acceptor close.
        Log_Prob expected_delta = -2.0 * log(0.4);
        CHECK("HSMM path drops two intron self-loop transition costs",
              fabs((hsmm - geometric) - expected_delta) < 1e-9);
        CHECK("duration model changes path score vs geometric",
              fabs(hsmm - geometric) > 1e-6);
    }

    static void test_viterbi_duration_prefers_length_mode() {
        cout << "\n[TEST 8] Duration on vs off changes MAP path under length pressure\n";

        vector<Nucleotide> nucs = {
            Nucleotide::G, Nucleotide::T,
            Nucleotide::A, Nucleotide::A, Nucleotide::A, Nucleotide::A,
            Nucleotide::A, Nucleotide::G
        };
        Emission_Model model = make_viterbi_test_model(nucs.size());
        auto transitions = make_viterbi_log_zero_matrix();
        transitions[idx(State::START)][idx(State::DONOR_1)] = 0.0;
        transitions[idx(State::DONOR_1)][idx(State::INTRON_1)] = 0.0;
        transitions[idx(State::INTRON_1)][idx(State::INTRON_1)] = 0.0;
        transitions[idx(State::INTRON_1)][idx(State::ACCEPTOR_1)] = 0.0;
        transitions[idx(State::ACCEPTOR_1)][idx(State::END)] = 0.0;
        transitions[idx(State::START)][idx(State::INTERGENIC)] = log(0.01);
        transitions[idx(State::INTERGENIC)][idx(State::INTERGENIC)] = 0.0;
        transitions[idx(State::INTERGENIC)][idx(State::END)] = 0.0;

        vector<Log_Prob> prefer_mid(8, LOG_ZERO);
        prefer_mid[4] = 0.0;  // log 1 at duration 4

        vector<State> geometric_path = Viterbi::decode(
            nucs, transitions, model, 1, 7, 0.0, {});
        vector<State> duration_path = Viterbi::decode(
            nucs, transitions, model, 1, 7, 0.0, prefer_mid);

        CHECK("geometric decode returns a path", geometric_path.size() == nucs.size());
        CHECK("duration decode returns a path", duration_path.size() == nucs.size());

        Log_Prob geo_as_hsmm = Viterbi::path_log_prob(
            geometric_path, nucs, transitions, model, 0.0, 0, geometric_path.size(), prefer_mid);
        Log_Prob dur_as_hsmm = Viterbi::path_log_prob(
            duration_path, nucs, transitions, model, 0.0, 0, duration_path.size(), prefer_mid);
        CHECK("duration MAP path scores at least as high under HSMM as geometric MAP",
              dur_as_hsmm + 1e-9 >= geo_as_hsmm);
    }

    static void run_Viterbi_tests() {
        cout << "\nRunning Viterbi tests...\n";

        test_viterbi_empty_sequence();
        test_viterbi_deterministic_start_codon_path();
        test_viterbi_intergenic_self_loop_path();
        test_viterbi_chooses_higher_scoring_path();
        test_viterbi_intron_body_min_length();
        test_viterbi_gene_start_penalty();
        test_viterbi_path_log_prob_matches_duration_decode_score();
        test_viterbi_duration_prefers_length_mode();

        cout << "\nDone.\n";
    }

}
