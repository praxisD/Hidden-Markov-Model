#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace hmm {

using Probability = double;
using ProbabilityMatrix = std::vector<std::vector<Probability>>;

class HiddenMarkovModel {
public:
    HiddenMarkovModel(std::vector<std::string> states,
                      std::vector<std::string> observations,
                      std::vector<Probability> startProbabilities,
                      ProbabilityMatrix transitionProbabilities,
                      ProbabilityMatrix emissionProbabilities);

    [[nodiscard]] const std::vector<std::string>& states() const;
    [[nodiscard]] const std::vector<std::string>& observations() const;

    [[nodiscard]] Probability startProbability(std::size_t state) const;
    [[nodiscard]] Probability transitionProbability(std::size_t fromState,
                                                    std::size_t toState) const;
    [[nodiscard]] Probability emissionProbability(std::size_t state,
                                                  std::size_t observation) const;
    [[nodiscard]] ProbabilityMatrix forward(
        const std::vector<std::size_t>& observationSequence) const;

private:
    std::vector<std::string> states_;
    std::vector<std::string> observations_;
    std::vector<Probability> startProbabilities_;
    ProbabilityMatrix transitionProbabilities_;
    ProbabilityMatrix emissionProbabilities_;

    void validate() const;
};

} // namespace hmm
