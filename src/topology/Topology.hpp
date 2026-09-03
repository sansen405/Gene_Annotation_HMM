#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <limits>
#include <unordered_map>
#include <string>

namespace gene_hmm {

    using namespace std;

    using Log_Prob = double;
    const Log_Prob LOG_INF = numeric_limits<Log_Prob>::infinity();
    const Log_Prob LOG_ZERO = -numeric_limits<Log_Prob>::infinity();

    enum class Nucleotide: uint8_t{
        A = 1,
        C = 2,
        G = 3,
        T = 4,
    };

    enum class State: uint8_t{
        START = 0, 
        INTERGENIC = 1,
        START_CODON_1 = 2, START_CODON_2 = 3, START_CODON_3 = 4,
        EXON_FRAME_1 = 5, EXON_FRAME_2 = 6, EXON_FRAME_3 = 7,
        DONOR_1 = 8, DONOR_2 = 9, DONOR_3 = 10,
        INTRON_1 = 11, INTRON_2 = 12, INTRON_3 = 13,
        ACCEPTOR_1 = 14, ACCEPTOR_2 = 15, ACCEPTOR_3 = 16,
        STOP_CODON_1 = 17, STOP_CODON_2 = 18, STOP_CODON_3 = 19,
        END = 20,
    };

    const size_t NUM_STATES = 21;
    const size_t NUM_NUCLEOTIDES = 4;
    const size_t MEMORY_WINDOW = 5;

    inline size_t idx(State s)      { return static_cast<size_t>(s); }
    inline size_t idx(Nucleotide n) { return static_cast<size_t>(n) - 1; }
    inline State  st(size_t s)      { return static_cast<State>(s); }

    const unordered_map<State, vector<State>> Transitions = {
        {State::START, {State::INTERGENIC, State::START_CODON_1}},
        {State::INTERGENIC, {State::INTERGENIC, State::START_CODON_1, State::END}},
        {State::START_CODON_1, {State::START_CODON_2}},
        {State::START_CODON_2, {State::START_CODON_3}},
        {State::START_CODON_3, {State::EXON_FRAME_1, State::DONOR_1}},
        {State::EXON_FRAME_1, {State::EXON_FRAME_2, State::DONOR_2}},
        {State::EXON_FRAME_2, {State::EXON_FRAME_3, State::DONOR_3}},
        {State::EXON_FRAME_3, {State::EXON_FRAME_1, State::DONOR_1, State::STOP_CODON_1}},
        {State::DONOR_1, {State::INTRON_1}},
        {State::DONOR_2, {State::INTRON_2}},
        {State::DONOR_3, {State::INTRON_3}},
        {State::INTRON_1, {State::INTRON_1, State::ACCEPTOR_1}},
        {State::INTRON_2, {State::INTRON_2, State::ACCEPTOR_2}},
        {State::INTRON_3, {State::INTRON_3, State::ACCEPTOR_3}},
        {State::ACCEPTOR_1, {State::EXON_FRAME_1}},
        {State::ACCEPTOR_2, {State::EXON_FRAME_2}},
        {State::ACCEPTOR_3, {State::EXON_FRAME_3}},
        {State::STOP_CODON_1, {State::STOP_CODON_2}},
        {State::STOP_CODON_2, {State::STOP_CODON_3}},
        {State::STOP_CODON_3, {State::INTERGENIC, State::END}},
        {State::END, {}}
    };

    // Emitting-state adjacency for sparse DP (excludes START/END).
    // Average degree ~1.5 vs dense S≈19 emitting states.
    inline const array<vector<State>, NUM_STATES>& emitting_predecessors() {
        static const array<vector<State>, NUM_STATES> preds = [] {
            array<vector<State>, NUM_STATES> built{};
            for (const auto& [from, tos] : Transitions) {
                if (from == State::START || from == State::END) continue;
                for (State to : tos) {
                    if (to == State::START || to == State::END) continue;
                    built[idx(to)].push_back(from);
                }
            }
            return built;
        }();
        return preds;
    }

    inline const array<vector<State>, NUM_STATES>& emitting_successors() {
        static const array<vector<State>, NUM_STATES> succs = [] {
            array<vector<State>, NUM_STATES> built{};
            for (const auto& [from, tos] : Transitions) {
                if (from == State::START || from == State::END) continue;
                for (State to : tos) {
                    if (to == State::START || to == State::END) continue;
                    built[idx(from)].push_back(to);
                }
            }
            return built;
        }();
        return succs;
    }
}