#include "hmm/HiddenMarkovModel.hpp"

#include <stdexcept>

namespace hmm {

HiddenMarkovModel::HiddenMarkovModel(
    std::vector<std::string> states,
    std::vector<std::string> observations,
    std::vector<Probability> startProbabilities,
    ProbabilityMatrix transitionProbabilities,
    ProbabilityMatrix emissionProbabilities)
    : states_(std::move(states)),
      observations_(std::move(observations)),
      startProbabilities_(std::move(startProbabilities)),
      transitionProbabilities_(std::move(transitionProbabilities)),
      emissionProbabilities_(std::move(emissionProbabilities)) {
    validate();
}

const std::vector<std::string>& HiddenMarkovModel::states() const {
    return states_;
}

const std::vector<std::string>& HiddenMarkovModel::observations() const {
    return observations_;
}

Probability HiddenMarkovModel::startProbability(std::size_t state) const {
    return startProbabilities_.at(state);
}

Probability HiddenMarkovModel::transitionProbability(std::size_t fromState,
                                                     std::size_t toState) const {
    return transitionProbabilities_.at(fromState).at(toState);
}

Probability HiddenMarkovModel::emissionProbability(std::size_t state,
                                                   std::size_t observation) const {
    return emissionProbabilities_.at(state).at(observation);
}

void HiddenMarkovModel::validate() const {
    if (states_.empty()) {
        throw std::invalid_argument("HiddenMarkovModel requires at least one state");
    }

    if (observations_.empty()) {
        throw std::invalid_argument("HiddenMarkovModel requires at least one observation");
    }

    if (startProbabilities_.size() != states_.size()) {
        throw std::invalid_argument("Start probabilities must match the number of states");
    }

    if (transitionProbabilities_.size() != states_.size()) {
        throw std::invalid_argument("Transition matrix must have one row per state");
    }

    for (const auto& row : transitionProbabilities_) {
        if (row.size() != states_.size()) {
            throw std::invalid_argument("Transition matrix must be state_count x state_count");
        }
    }

    if (emissionProbabilities_.size() != states_.size()) {
        throw std::invalid_argument("Emission matrix must have one row per state");
    }

    for (const auto& row : emissionProbabilities_) {
        if (row.size() != observations_.size()) {
            throw std::invalid_argument("Emission matrix rows must match the observation count");
        }
    }
}

} // namespace hmm
