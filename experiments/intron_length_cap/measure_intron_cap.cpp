// Diagnostic: how much intron recall is the hard p95 intron-length cap costing?
//
// Reproduces the decoder's intron-length-cap construction from
// validation/full_genome_validation.cpp: the cap = the Pth percentile of
// TRAIN intron-body-state run lengths, and the Viterbi decoder forbids any
// intron body longer than the cap (LOG_ZERO beyond it). This tool measures the
// fraction of held-out (test) GOLD intron bodies whose length exceeds that cap.
// Those introns are structurally impossible to decode, so 1 - that fraction is
// an upper bound (ceiling) on intron recall imposed purely by the cap.

#include "../../src/genome_profiles/Genome_Profile.hpp"
#include "../../src/parsers/FNA_Parser.hpp"
#include "../../src/parsers/GFF_Parser.hpp"
#include "../../src/topology/Topology.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace gene_hmm;
using namespace std;

namespace {

struct Sequence_Data {
    vector<State> states;
    vector<int> regions;
    vector<Chromosome_Range> chromosomes;
};

bool is_intron_body(State state) {
    return state == State::INTRON_1 ||
           state == State::INTRON_2 ||
           state == State::INTRON_3;
}

bool is_usable_region(int region) {
    return region != GFF_Parser::IGNORED_REGION;
}

void append_dataset(
    Sequence_Data& combined,
    const string& name,
    const string& fasta_path,
    const string& gff_path)
{
    size_t offset = combined.states.size();

    vector<Nucleotide> nucleotides = FNA_Parser::parse_sequence(fasta_path);
    vector<Chromosome_Range> chromosomes = FNA_Parser::get_chromosome_ranges(fasta_path);
    string mutable_gff_path = gff_path;
    string mutable_fasta_path = fasta_path;
    vector<int> regions = GFF_Parser::parse_regions(mutable_gff_path, mutable_fasta_path);
    vector<State> states = GFF_Parser::parse_states(regions);

    if (states.size() != nucleotides.size()) {
        throw runtime_error("State length does not match nucleotide length for " + name + ".");
    }

    combined.states.insert(combined.states.end(), states.begin(), states.end());
    combined.regions.insert(combined.regions.end(), regions.begin(), regions.end());
    for (auto range : chromosomes) {
        range.name = name + ":" + range.name;
        range.start += offset;
        range.end += offset;
        combined.chromosomes.push_back(range);
    }
}

vector<Chromosome_Range> split_usable_ranges(
    const vector<Chromosome_Range>& ranges,
    const vector<int>& regions)
{
    vector<Chromosome_Range> usable_ranges;
    for (const auto& range : ranges) {
        size_t pos = range.start;
        while (pos < range.end) {
            while (pos < range.end && !is_usable_region(regions[pos])) {
                pos++;
            }
            if (pos >= range.end) {
                break;
            }
            size_t start = pos;
            while (pos < range.end && is_usable_region(regions[pos])) {
                pos++;
            }
            usable_ranges.push_back({range.name, start, pos});
        }
    }
    return usable_ranges;
}

vector<size_t> collect_intron_body_lengths(
    const vector<State>& states,
    const vector<Chromosome_Range>& ranges)
{
    vector<size_t> lengths;
    for (const auto& range : ranges) {
        size_t i = range.start;
        while (i < range.end) {
            if (!is_intron_body(states[i])) {
                i++;
                continue;
            }
            size_t start = i;
            while (i < range.end && is_intron_body(states[i])) {
                i++;
            }
            lengths.push_back(i - start);
        }
    }
    return lengths;
}

size_t percentile_length(vector<size_t> lengths, double percentile) {
    if (lengths.empty()) {
        return numeric_limits<size_t>::max();
    }
    sort(lengths.begin(), lengths.end());
    size_t index = static_cast<size_t>(percentile * static_cast<double>(lengths.size() - 1));
    return lengths[index];
}

size_t count_exceeding(const vector<size_t>& lengths, size_t cap) {
    size_t n = 0;
    for (size_t l : lengths) {
        if (l > cap) {
            n++;
        }
    }
    return n;
}

double fraction_exceeding(const vector<size_t>& lengths, size_t cap) {
    if (lengths.empty()) {
        return 0.0;
    }
    return static_cast<double>(count_exceeding(lengths, cap)) / static_cast<double>(lengths.size());
}

string value_after_arg(int argc, char** argv, const string& name, const string& fallback) {
    for (int i = 1; i + 1 < argc; i++) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

} // namespace

int main(int argc, char** argv) {
    try {
        string profile_path = value_after_arg(argc, argv, "--profile", "");
        if (profile_path.empty()) {
            cerr << "Usage: " << argv[0] << " --profile PATH [--cap-percentile 0.95]\n";
            return 1;
        }
        double cap_percentile = stod(value_after_arg(argc, argv, "--cap-percentile", "0.95"));

        Genome_Profile loaded = Genome_Profile::load(profile_path);

        // Pool train data (defines the cap, exactly like the decoder) and test data.
        Sequence_Data train;
        Sequence_Data test;
        for (const auto& dataset : loaded.species) {
            append_dataset(train, dataset.name, dataset.train_fasta_path, dataset.train_gff_path);
            append_dataset(test, dataset.name, dataset.test_fasta_path, dataset.test_gff_path);
        }

        vector<Chromosome_Range> train_ranges = split_usable_ranges(train.chromosomes, train.regions);
        vector<Chromosome_Range> test_ranges = split_usable_ranges(test.chromosomes, test.regions);

        vector<size_t> train_lengths = collect_intron_body_lengths(train.states, train_ranges);
        vector<size_t> test_lengths = collect_intron_body_lengths(test.states, test_ranges);

        size_t cap = percentile_length(train_lengths, cap_percentile);

        cout << fixed << setprecision(4);
        cout << "=== Intron-length-cap diagnostic ===\n";
        cout << "Profile: " << loaded.name << "\n";
        cout << "Species: " << loaded.species.size() << "\n\n";

        cout << "Train intron bodies (define the cap): " << train_lengths.size() << "\n";
        cout << "Test  intron bodies (gold, held out): " << test_lengths.size() << "\n\n";

        auto pct = [](const vector<size_t>& v, double p) { return percentile_length(v, p); };
        cout << "Train intron-body length percentiles (bp):\n";
        cout << "  p50=" << pct(train_lengths, 0.50)
             << "  p90=" << pct(train_lengths, 0.90)
             << "  p95=" << pct(train_lengths, 0.95)
             << "  p99=" << pct(train_lengths, 0.99)
             << "  p99.9=" << pct(train_lengths, 0.999)
             << "  max=" << pct(train_lengths, 1.0) << "\n";
        cout << "Test  intron-body length percentiles (bp):\n";
        cout << "  p50=" << pct(test_lengths, 0.50)
             << "  p90=" << pct(test_lengths, 0.90)
             << "  p95=" << pct(test_lengths, 0.95)
             << "  p99=" << pct(test_lengths, 0.99)
             << "  p99.9=" << pct(test_lengths, 0.999)
             << "  max=" << pct(test_lengths, 1.0) << "\n\n";

        cout << "Current cap = train p" << (cap_percentile * 100.0) << " = " << cap << " bp\n\n";

        size_t over = count_exceeding(test_lengths, cap);
        double frac = fraction_exceeding(test_lengths, cap);
        cout << "Test gold introns longer than the cap: " << over
             << " / " << test_lengths.size()
             << "  (" << (frac * 100.0) << "%)\n";
        cout << "==> Intron-recall CEILING imposed by the cap: "
             << ((1.0 - frac) * 100.0) << "%\n\n";

        cout << "Recall ceiling under alternative caps (from train distribution):\n";
        cout << left << setw(14) << "cap source" << setw(10) << "cap(bp)"
             << setw(16) << "test over-cap" << setw(16) << "recall ceiling" << "\n";
        struct Option { string label; size_t cap; };
        vector<Option> options = {
            {"train p95", pct(train_lengths, 0.95)},
            {"train p99", pct(train_lengths, 0.99)},
            {"train p99.9", pct(train_lengths, 0.999)},
            {"train max", pct(train_lengths, 1.0)},
        };
        for (const auto& opt : options) {
            double f = fraction_exceeding(test_lengths, opt.cap);
            cout << left << setw(14) << opt.label
                 << setw(10) << opt.cap
                 << setw(16) << count_exceeding(test_lengths, opt.cap)
                 << setw(16) << ((1.0 - f) * 100.0) << "\n";
        }
        cout << "\n";

        // Per-species view (test only): cap is global, lengths are per species.
        cout << "Per-species test intron bodies over the current cap (" << cap << " bp):\n";
        cout << left << setw(20) << "species" << setw(10) << "introns"
             << setw(12) << "over-cap" << setw(14) << "over-cap %"
             << setw(12) << "test max" << "\n";
        for (const auto& dataset : loaded.species) {
            Sequence_Data sp;
            append_dataset(sp, dataset.name, dataset.test_fasta_path, dataset.test_gff_path);
            vector<Chromosome_Range> sp_ranges = split_usable_ranges(sp.chromosomes, sp.regions);
            vector<size_t> sp_lengths = collect_intron_body_lengths(sp.states, sp_ranges);
            if (sp_lengths.empty()) {
                cout << left << setw(20) << dataset.name << setw(10) << 0
                     << setw(12) << 0 << setw(14) << "n/a" << setw(12) << 0 << "\n";
                continue;
            }
            size_t sp_over = count_exceeding(sp_lengths, cap);
            cout << left << setw(20) << dataset.name
                 << setw(10) << sp_lengths.size()
                 << setw(12) << sp_over
                 << setw(14) << (static_cast<double>(sp_over) / sp_lengths.size() * 100.0)
                 << setw(12) << pct(sp_lengths, 1.0) << "\n";
        }

        return 0;
    } catch (const exception& error) {
        cerr << error.what() << "\n";
        return 1;
    }
}
