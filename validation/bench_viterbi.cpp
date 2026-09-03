#include "../src/decoding/Viterbi.hpp"
#include "../src/genome_profiles/Genome_Profile.hpp"
#include "../src/model/emission/Emission_Model.hpp"
#include "../src/model/transition/Transition_Model.hpp"
#include "../src/parsers/FNA_Parser.hpp"
#include "../src/parsers/GFF_Parser.hpp"
#include "../src/topology/Topology.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace gene_hmm;
using namespace std;
using Clock = chrono::steady_clock;

size_t sparse_edge_count() {
    size_t edges = 0;
    for (State s = State::INTERGENIC; s < State::END; s = st(idx(s) + 1)) {
        edges += emitting_predecessors()[idx(s)].size();
    }
    return edges;
}

Emission_Model train_emissions(
    const vector<State>& states,
    const vector<Nucleotide>& nucleotides,
    const vector<Chromosome_Range>& train_ranges)
{
    Emission_Model model;
    model.intergenic_lp = Emission_Model::compute_markov1_log_probs(
        Emission_Model::count_markov1_emissions(
            states, nucleotides, train_ranges, {State::INTERGENIC}));
    model.intron_lp = Emission_Model::compute_markov5_log_probs(
        Emission_Model::count_markov5_emissions(
            states, nucleotides, train_ranges,
            {State::INTRON_1, State::INTRON_2, State::INTRON_3}));
    for (size_t frame = 0; frame < 3; ++frame) {
        State exon = st(idx(State::EXON_FRAME_1) + frame);
        model.exon_frame_lp[frame] = Emission_Model::compute_markov5_log_probs(
            Emission_Model::count_markov5_emissions(
                states, nucleotides, train_ranges, {exon}));
    }
    return model;
}

}  // namespace

int main(int argc, char** argv) {
    string profile_path = "src/genome_profiles/fission_yeasts/fission_yeasts.json";
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "--profile") && i + 1 < argc) {
            profile_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            cerr << "Usage: " << argv[0] << " [--profile PATH]\n";
            return 0;
        }
    }

    profile = Genome_Profile::load(profile_path);
    if (profile.species.empty()) {
        cerr << "Profile has no species entries\n";
        return 1;
    }

    const auto& species = profile.species.front();

    string train_fasta = species.train_fasta_path;
    string train_gff = species.train_gff_path;
    vector<Nucleotide> train_nucs = FNA_Parser::parse_sequence(train_fasta);
    vector<Chromosome_Range> train_chrs = FNA_Parser::get_chromosome_ranges(train_fasta);
    vector<int> train_regions = GFF_Parser::parse_regions(train_gff, train_fasta);
    vector<State> train_states = GFF_Parser::parse_states(train_regions);

    auto transition_lp = Transition_Model::compute_log_probs(train_states, train_chrs);
    Emission_Model emission = train_emissions(train_states, train_nucs, train_chrs);

    string test_fasta = species.test_fasta_path;
    vector<Nucleotide> nucleotides = FNA_Parser::parse_sequence(test_fasta);
    vector<Chromosome_Range> chromosomes = FNA_Parser::get_chromosome_ranges(test_fasta);
    if (chromosomes.empty()) {
        cerr << "No chromosomes in test FASTA\n";
        return 1;
    }

    if (!profile.splice_cnn.test_score_paths.empty()) {
        emission.load_splice_cnn_scores(profile.splice_cnn.test_score_paths.front(), nucleotides.size());
    }
    if (!profile.start_cnn.test_score_paths.empty()) {
        emission.load_start_cnn_scores(profile.start_cnn.test_score_paths.front(), nucleotides.size());
    }
    emission.set_splice_cnn_calibration(
        profile.splice_cnn.donor_scale,
        profile.splice_cnn.donor_bias,
        profile.splice_cnn.acceptor_scale,
        profile.splice_cnn.acceptor_bias);
    emission.set_start_cnn_calibration(
        profile.start_cnn.start_scale,
        profile.start_cnn.start_bias);

    const auto& chr = chromosomes.front();
    const size_t T = chr.end - chr.start;
    vector<Nucleotide> slice(nucleotides.begin() + chr.start, nucleotides.begin() + chr.end);

    const size_t S_emit = static_cast<size_t>(State::END) - static_cast<size_t>(State::INTERGENIC);
    const size_t E = sparse_edge_count();

    cout << fixed << setprecision(4);
    cout << "=== Viterbi performance note ===\n";
    cout << "Species:              " << species.name << "\n";
    cout << "Chromosome:           " << chr.name << "\n";
    cout << "Length T:             " << T << " bases\n";
    cout << "Emitting states S:    " << S_emit << "\n";
    cout << "Sparse legal edges E: " << E
         << " (dense would be " << (S_emit * S_emit) << ")\n";
    cout << "Asymptotics:          O(T · E) time with sparse Transitions; "
            "O(T · S) backpointer memory (scores/lengths are 2-row streaming)\n";
    const double backpointer_mib =
        (static_cast<double>(T) * NUM_STATES * sizeof(State)) / (1024.0 * 1024.0);
    const double legacy_gib =
        (static_cast<double>(T) * NUM_STATES * (sizeof(Log_Prob) + sizeof(State) + sizeof(size_t)))
        / (1024.0 * 1024.0 * 1024.0);
    cout << "Streaming DP tables:  ~" << backpointer_mib << " MiB backpointers "
            "(legacy full V+B+len ≈ " << legacy_gib << " GiB)\n";

    auto t0 = Clock::now();
    auto path = Viterbi::decode(slice, transition_lp, emission, 20, 196, 1.0);
    auto t1 = Clock::now();
    double seconds = chrono::duration<double>(t1 - t0).count();
    double mbps = (T / 1e6) / seconds;

    cout << "Wall-clock decode:    " << seconds << " s\n";
    cout << "Throughput:           " << mbps << " Mbp/s\n";
    cout << "Path length:          " << path.size() << "\n";
    return 0;
}
