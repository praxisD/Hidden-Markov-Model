#include "hmm/HiddenMarkovModel.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

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

ProbabilityTensor HiddenMarkovModel::transitionResponsibility(
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
    ProbabilityTensor xi(
        timeSteps - 1,
        std::vector<std::vector<Probability>>(
            stateCount,
            std::vector<Probability>(stateCount, 0.0)));

    for (std::size_t time = 0; time < timeSteps - 1; ++time) {
        for (std::size_t fromState = 0; fromState < stateCount; ++fromState) {
            for (std::size_t toState = 0; toState < stateCount; ++toState) {
                xi[time][fromState][toState] =
                    (alpha[time][fromState] *
                     transitionProbabilities_[fromState][toState] *
                     emissionProbabilities_[toState][observationSequence[time + 1]] *
                     beta[time + 1][toState]) / totalProbability;
            }
        }
    }

    return xi;
}

BaumWelchResult HiddenMarkovModel::baumWelch(
    const std::vector<std::size_t>& observationSequence,
    std::size_t maxIterations,
    Probability tolerance) {
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
    const auto observationCount = observations_.size();
    const auto timeSteps = observationSequence.size();

    Probability previousProbability = 0.0;
    Probability currentProbability = sequenceProbability(observationSequence);

    if (maxIterations == 0) {
        return {0, currentProbability};
    }

    for (std::size_t iteration = 0; iteration < maxIterations; ++iteration) {
        const auto gamma = stateResponsibility(observationSequence);
        const auto xi = transitionResponsibility(observationSequence);

        // Update start probabilities
        for (std::size_t state = 0; state < stateCount; ++state) {
            startProbabilities_[state] = gamma[0][state];
        }

        // Update transition probabilities
        for (std::size_t fromState = 0; fromState < stateCount; ++fromState) {
            Probability sumGammaFromState = 0.0;

            for (std::size_t time = 0; time < timeSteps - 1; ++time) {
                sumGammaFromState += gamma[time][fromState];
            }

            if (sumGammaFromState == 0.0) {
                continue;
            }

            for (std::size_t toState = 0; toState < stateCount; ++toState) {
                Probability sumXiFromToState = 0.0;

                for (std::size_t time = 0; time < timeSteps - 1; ++time) {
                    sumXiFromToState += xi[time][fromState][toState];
                }

                transitionProbabilities_[fromState][toState] =
                    sumXiFromToState / sumGammaFromState;
            }
        }

        // Update emission probabilities
        for (std::size_t state = 0; state < stateCount; ++state) {            
            Probability sumGammaForState = 0.0;
            
            for (std::size_t time = 0; time < timeSteps; ++time) {
                sumGammaForState += gamma[time][state];
            }

            if (sumGammaForState == 0.0) {
                continue;
            }

            for (std::size_t observation = 0; observation < observationCount; ++observation) {

                Probability sumGammaObservationForState = 0.0;

                for (std::size_t time = 0; time < timeSteps; ++time) {
                    if (observationSequence[time] == observation) {
                        sumGammaObservationForState += gamma[time][state];
                    }
                }

                emissionProbabilities_[state][observation] =
                    sumGammaObservationForState / sumGammaForState;
            }
        }

        previousProbability = currentProbability;
        currentProbability = sequenceProbability(observationSequence);

        if (std::abs(currentProbability - previousProbability) < tolerance) {
            return {iteration + 1, currentProbability};
        }
    }

    return {maxIterations, currentProbability};
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
