cmake_minimum_required(VERSION 3.16)
project(gene_annotation_hmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
  # Fall back to include-path discovery (Homebrew / apt header-only install).
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS
      $ENV{JSON_INCLUDE}
      /opt/homebrew/include
      /usr/local/include
      /usr/include
  )
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
      "nlohmann/json.hpp not found. Install nlohmann-json or set JSON_INCLUDE.")
  endif()
endif()

set(COMMON_SOURCES
  src/decoding/Viterbi.cpp
  src/model/transition/Transition_Model.cpp
  src/genome_profiles/Genome_Profile.cpp
  src/parsers/FNA_Parser.cpp
  src/parsers/GFF_Parser.cpp
  src/model/emission/Emission_Model.cpp
  src/model/cnn/splice/Splice_CNN_Scores.cpp
  src/model/cnn/start/Start_CNN_Scores.cpp
)

function(gene_hmm_target name)
  add_executable(${name} ${ARGN} ${COMMON_SOURCES})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  if(nlohmann_json_FOUND)
    target_link_libraries(${name} PRIVATE nlohmann_json::nlohmann_json)
  else()
    target_include_directories(${name} PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
  endif()
endfunction()

gene_hmm_target(gene_hmm_tests
  src/main.cpp
  src/decoding/Forward_Backward.cpp
)

gene_hmm_target(full_genome_validation
  validation/full_genome_validation.cpp
)

gene_hmm_target(hmm_predict_fna
  src/tools/predict_fna.cpp
  src/decoding/Forward_Backward.cpp
)

gene_hmm_target(train_hmm_matrices
  src/model/training_pipeline/train_hmm_matrices.cpp
)

gene_hmm_target(bench_viterbi
  validation/bench_viterbi.cpp
)
