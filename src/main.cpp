#include "hmm/HiddenMarkovModel.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::vector<std::string> states{"Healthy", "Fever"};
    const std::vector<std::string> observations{"Normal", "Cold", "Dizzy"};

    const std::vector<hmm::Probability> startProbabilities{0.6, 0.4};

    const hmm::ProbabilityMatrix transitionProbabilities{
        {0.7, 0.3},
        {0.4, 0.6},
    };

    const hmm::ProbabilityMatrix emissionProbabilities{
        {0.5, 0.4, 0.1},
        {0.1, 0.3, 0.6},
    };

    const hmm::HiddenMarkovModel model{
        states,
        observations,
        startProbabilities,
        transitionProbabilities,
        emissionProbabilities,
    };

    std::cout << "Hidden Markov Model starter\n";
    std::cout << "States: ";
    for (const auto& state : model.states()) {
        std::cout << state << ' ';
    }
    std::cout << '\n';

    std::cout << "P(Fever | Healthy): "
              << model.transitionProbability(0, 1) << '\n';
    std::cout << "P(Dizzy | Fever): "
              << model.emissionProbability(1, 2) << '\n';

    return 0;
}
