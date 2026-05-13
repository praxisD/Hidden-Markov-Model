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

ProbabilityMatrix HiddenMarkovModel::forward(
    const std::vector<std::size_t>& observationSequence) const {
    if (observationSequence.empty()) {
        throw std::invalid_argument("Observation sequence must not be empty");
    }

    for (const auto observation : observationSequence) {
        if (observation >= observations_.size()) {
            throw std::invalid_argument(
                "Observation sequence contains an invalid observation index");
        }
    }

    const auto stateCount = states_.size();
    const auto timeSteps = observationSequence.size();
    ProbabilityMatrix alpha(timeSteps, std::vector<Probability>(stateCount, 0.0));

    for (std::size_t state = 0; state < stateCount; ++state) {
        alpha[0][state] = startProbabilities_[state] *
                          emissionProbabilities_[state][observationSequence[0]];
    }

    for (std::size_t time = 1; time < timeSteps; ++time) {
        for (std::size_t toState = 0; toState < stateCount; ++toState) {
            Probability previousSum = 0.0;

            for (std::size_t fromState = 0; fromState < stateCount; ++fromState) {
                previousSum += alpha[time - 1][fromState] *
                               transitionProbabilities_[fromState][toState];
            }

            alpha[time][toState] =
                emissionProbabilities_[toState][observationSequence[time]] *
                previousSum;
        }
    }

    return alpha;
}

ProbabilityMatrix HiddenMarkovModel::backward(
    const std::vector<std::size_t>& observationSequence) const {
    if (observationSequence.empty()) {
        throw std::invalid_argument("Observation sequence must not be empty");
    }

    for (const auto observation : observationSequence) {
        if (observation >= observations_.size()) {
            throw std::invalid_argument(
                "Observation sequence contains an invalid observation index");
        }
    }

    const auto stateCount = states_.size();
    const auto timeSteps = observationSequence.size();
    ProbabilityMatrix beta(timeSteps, std::vector<Probability>(stateCount, 0.0));

    for (std::size_t state = 0; state < stateCount; ++state) {
        beta[timeSteps - 1][state] = 1.0;
    }

    for (std::size_t time = timeSteps - 1; time-- > 0;) {
        for (std::size_t fromState = 0; fromState < stateCount; ++fromState) {
            Probability nextSum = 0.0;

            for (std::size_t toState = 0; toState < stateCount; ++toState) {
                nextSum += transitionProbabilities_[fromState][toState] *
                           emissionProbabilities_[toState][observationSequence[time + 1]] *
                           beta[time + 1][toState];
            }

            beta[time][fromState] = nextSum;
        }
    }

    return beta;
}

Probability HiddenMarkovModel::sequenceProbability(
    const std::vector<std::size_t>& observationSequence) const {
    const auto alpha = forward(observationSequence);
    Probability totalProbability = 0.0;

    for (const auto& prob : alpha.back()) {
        totalProbability += prob;
    }

    return totalProbability;
}

ProbabilityMatrix HiddenMarkovModel::stateResponsibility(
    const std::vector<std::size_t>& observationSequence) const {
    const auto alpha = forward(observationSequence);
    const auto beta = backward(observationSequence);

    Probability totalProbability = 0.0;
    for (const auto probability : alpha.back()) {
        totalProbability += probability;
    }

    if (totalProbability == 0.0) {
        throw std::runtime_error("Total probability of the observation sequence is zero");
    }

    const auto stateCount = states_.size();
    const auto timeSteps = observationSequence.size();
    ProbabilityMatrix gamma(timeSteps, std::vector<Probability>(stateCount, 0.0));

    for (std::size_t time = 0; time < timeSteps; ++time) {
        for (std::size_t state = 0; state < stateCount; ++state) {
            gamma[time][state] = (alpha[time][state] * beta[time][state]) / totalProbability;
        }
    }

    return gamma;
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
