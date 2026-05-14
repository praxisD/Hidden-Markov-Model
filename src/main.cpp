#include "hmm/HiddenMarkovModel.hpp"
#include "hmm/Inputs.hpp"

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void printStartProbabilities(const hmm::HiddenMarkovModel& model) {
    std::cout << "Initial probabilities:\n";
    for (std::size_t state = 0; state < model.states().size(); ++state) {
        std::cout << "  " << model.states()[state] << ": "
                  << model.startProbability(state) << '\n';
    }
}

void printTransitionMatrix(const hmm::HiddenMarkovModel& model) {
    std::cout << "Transition matrix:\n";
    for (std::size_t fromState = 0; fromState < model.states().size(); ++fromState) {
        std::cout << "  " << model.states()[fromState] << ":";
        for (std::size_t toState = 0; toState < model.states().size(); ++toState) {
            std::cout << ' ' << model.transitionProbability(fromState, toState);
        }
        std::cout << '\n';
    }
}

void printEmissionMatrix(const hmm::HiddenMarkovModel& model) {
    std::cout << "Emission matrix:\n";
    for (std::size_t state = 0; state < model.states().size(); ++state) {
        std::cout << "  " << model.states()[state] << ":";
        for (std::size_t observation = 0;
             observation < model.observations().size();
             ++observation) {
            std::cout << ' ' << model.emissionProbability(state, observation);
        }
        std::cout << '\n';
    }
}

void printModelParameters(const hmm::HiddenMarkovModel& model) {
    printStartProbabilities(model);
    printTransitionMatrix(model);
    printEmissionMatrix(model);
}

} // namespace

int main() {
    try {
        const auto input = hmm::readTrainingInput(std::cin, std::cout);

        auto model = hmm::HiddenMarkovModel{
            hmm::makeIndexedNames("State", input.stateCount),
            hmm::makeIndexedNames("Observation", input.observationCount),
            hmm::equalProbabilities(input.stateCount),
            hmm::equalProbabilityMatrix(input.stateCount, input.stateCount),
            hmm::equalProbabilityMatrix(input.stateCount, input.observationCount),
        };

        std::cout << std::setprecision(8);
        std::cout << "\nLoaded " << input.observationSequence.size()
                  << " observations.\n\n";
        std::cout << "Parameters before training:\n";
        printModelParameters(model);

        const auto initialProbability =
            model.sequenceProbability(input.observationSequence);
        const auto result = model.baumWelch(input.observationSequence,
                                            input.maxIterations,
                                            input.tolerance);

        std::cout << "\nBaum-Welch completed after " << result.iterations
                  << " iteration";
        if (result.iterations != 1) {
            std::cout << 's';
        }
        std::cout << ".\n";
        std::cout << "Initial sequence probability: " << initialProbability << '\n';
        std::cout << "Final sequence probability: " << result.finalProbability << "\n\n";

        std::cout << "Parameters after training:\n";
        printModelParameters(model);

        std::cout << "\nNote: equal initialization keeps hidden states symmetric. "
                     "The trained rows may remain identical unless initialization is "
                     "made non-uniform later.\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
