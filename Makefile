# Gene Annotation HMM — portable build
# Requires: C++17 compiler, nlohmann/json (Homebrew, apt, or JSON_INCLUDE)

CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall
SRC_INC   = -Isrc

# Prefer Homebrew on macOS, then /usr/local, then pkg-config / system paths.
JSON_INCLUDE ?= $(shell \
	if [ -n "$$JSON_INCLUDE" ]; then echo "$$JSON_INCLUDE"; \
	elif [ -f /opt/homebrew/include/nlohmann/json.hpp ]; then echo /opt/homebrew/include; \
	elif [ -f /usr/local/include/nlohmann/json.hpp ]; then echo /usr/local/include; \
	elif [ -f /usr/include/nlohmann/json.hpp ]; then echo /usr/include; \
	elif pkg-config --exists nlohmann_json 2>/dev/null; then pkg-config --cflags-only-I nlohmann_json | sed 's/^-I//'; \
	else echo /usr/include; fi)

INCLUDES = $(SRC_INC) -I$(JSON_INCLUDE)

COMMON_SRCS = \
	src/decoding/Viterbi.cpp \
	src/model/transition/Transition_Model.cpp \
	src/genome_profiles/Genome_Profile.cpp \
	src/parsers/FNA_Parser.cpp \
	src/parsers/GFF_Parser.cpp \
	src/model/emission/Emission_Model.cpp \
	src/model/cnn/splice/Splice_CNN_Scores.cpp \
	src/model/cnn/start/Start_CNN_Scores.cpp

FB_SRCS = src/decoding/Forward_Backward.cpp

BUILD_DIR ?= build
PROFILE   ?= src/genome_profiles/fission_yeasts/fission_yeasts.json

.PHONY: all tests validation predict train-matrices clean help

all: tests validation predict train-matrices

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

tests: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		src/main.cpp $(COMMON_SRCS) $(FB_SRCS) \
		-o $(BUILD_DIR)/gene_hmm_tests

validation: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		validation/full_genome_validation.cpp $(COMMON_SRCS) \
		-o $(BUILD_DIR)/full_genome_validation

predict: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		src/tools/predict_fna.cpp $(COMMON_SRCS) $(FB_SRCS) \
		-o $(BUILD_DIR)/hmm_predict_fna

train-matrices: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		src/model/training_pipeline/train_hmm_matrices.cpp $(COMMON_SRCS) \
		-o $(BUILD_DIR)/train_hmm_matrices

run-tests: tests
	$(BUILD_DIR)/gene_hmm_tests $(PROFILE)

run-validation: validation
	$(BUILD_DIR)/full_genome_validation --profile $(PROFILE) \
		--results-dir validation/results/version_5

bench-viterbi: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		validation/bench_viterbi.cpp $(COMMON_SRCS) \
		-o $(BUILD_DIR)/bench_viterbi
	$(BUILD_DIR)/bench_viterbi --profile $(PROFILE)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets: all tests validation predict train-matrices run-tests run-validation bench-viterbi clean"
	@echo "Override: CXX=g++ JSON_INCLUDE=/path/to/include PROFILE=..."
