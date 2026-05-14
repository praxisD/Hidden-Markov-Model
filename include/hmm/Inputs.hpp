#pragma once

#include "hmm/HiddenMarkovModel.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace hmm {

struct TrainingInput {
    std::size_t stateCount;
    std::size_t observationCount;
    std::vector<std::size_t> observationSequence;
    std::size_t maxIterations;
    Probability tolerance;
};

TrainingInput readTrainingInput(std::istream& input, std::ostream& output);

std::vector<std::size_t> readObservationSequenceFile(
    const std::string& filePath,
    std::size_t observationCount);

std::vector<std::string> makeIndexedNames(const std::string& prefix,
                                          std::size_t count);

std::vector<Probability> equalProbabilities(std::size_t count);

ProbabilityMatrix equalProbabilityMatrix(std::size_t rows,
                                         std::size_t columns);

} // namespace hmm
