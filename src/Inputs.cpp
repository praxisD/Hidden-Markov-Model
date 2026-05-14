#include "hmm/Inputs.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hmm {
namespace {

constexpr std::size_t defaultMaxIterations = 100;
constexpr Probability defaultTolerance = 1e-6;
constexpr unsigned int defaultInitializationSeed = 12345;
constexpr Probability minInitialWeight = 0.5;
constexpr Probability maxInitialWeight = 1.5;

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos) {
        return {};
    }

    const auto last = value.find_last_not_of(" \t\n\r\f\v");
    return value.substr(first, last - first + 1);
}

bool containsOnlyDigits(const std::string& value) {
    return std::all_of(value.begin(), value.end(), [](const unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
}

std::size_t parseUnsignedInteger(const std::string& rawValue,
                                 const std::string& fieldName) {
    const auto value = trim(rawValue);
    if (value.empty()) {
        throw std::invalid_argument(fieldName + " is required");
    }

    if (!containsOnlyDigits(value)) {
        throw std::invalid_argument(fieldName + " must be a non-negative integer");
    }

    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(fieldName + " is too large");
    }

    if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(fieldName + " is too large");
    }

    return static_cast<std::size_t>(parsed);
}

std::size_t parsePositiveInteger(const std::string& rawValue,
                                 const std::string& fieldName) {
    const auto parsed = parseUnsignedInteger(rawValue, fieldName);
    if (parsed == 0) {
        throw std::invalid_argument(fieldName + " must be greater than zero");
    }

    return parsed;
}

Probability parsePositiveProbability(const std::string& rawValue,
                                      const std::string& fieldName) {
    const auto value = trim(rawValue);
    if (value.empty()) {
        throw std::invalid_argument(fieldName + " is required");
    }

    std::istringstream parser(value);
    Probability parsed = 0.0;
    if (!(parser >> parsed) || !std::isfinite(parsed) || parsed <= 0.0) {
        throw std::invalid_argument(fieldName + " must be a positive number");
    }

    parser >> std::ws;

    if (!parser.eof()) {
        throw std::invalid_argument(fieldName + " must be a positive number");
    }

    return parsed;
}

std::string readPromptLine(std::istream& input,
                           std::ostream& output,
                           const std::string& prompt) {
    output << prompt;
    output.flush();

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Expected more input");
    }

    return line;
}

std::size_t promptPositiveInteger(std::istream& input,
                                  std::ostream& output,
                                  const std::string& prompt,
                                  const std::string& fieldName) {
    while (true) {
        try {
            return parsePositiveInteger(readPromptLine(input, output, prompt),
                                        fieldName);
        } catch (const std::invalid_argument& error) {
            output << error.what() << '\n';
        }
    }
}

std::size_t promptPositiveIntegerWithDefault(std::istream& input,
                                             std::ostream& output,
                                             const std::string& prompt,
                                             const std::string& fieldName,
                                             std::size_t defaultValue) {
    while (true) {
        const auto line = trim(readPromptLine(input, output, prompt));
        if (line.empty()) {
            return defaultValue;
        }

        try {
            return parsePositiveInteger(line, fieldName);
        } catch (const std::invalid_argument& error) {
            output << error.what() << '\n';
        }
    }
}

Probability promptPositiveProbabilityWithDefault(std::istream& input,
                                                 std::ostream& output,
                                                 const std::string& prompt,
                                                 const std::string& fieldName,
                                                 Probability defaultValue) {
    while (true) {
        const auto line = trim(readPromptLine(input, output, prompt));
        if (line.empty()) {
            return defaultValue;
        }

        try {
            return parsePositiveProbability(line, fieldName);
        } catch (const std::invalid_argument& error) {
            output << error.what() << '\n';
        }
    }
}

std::string promptRequiredLine(std::istream& input,
                               std::ostream& output,
                               const std::string& prompt,
                               const std::string& fieldName) {
    while (true) {
        const auto line = trim(readPromptLine(input, output, prompt));
        if (!line.empty()) {
            return line;
        }

        output << fieldName << " is required\n";
    }
}

std::size_t parseObservationIndex(const std::string& token,
                                  std::size_t observationCount,
                                  std::size_t lineNumber) {
    const auto fieldName = "Observation token '" + token + "' on line " +
                           std::to_string(lineNumber);
    const auto index = parseUnsignedInteger(token, fieldName);

    if (index >= observationCount) {
        throw std::invalid_argument(
            fieldName + " is outside the valid range 0-" +
            std::to_string(observationCount - 1));
    }

    return index;
}

std::vector<Probability> randomProbabilities(std::size_t count,
                                             std::mt19937& generator) {
    if (count == 0) {
        throw std::invalid_argument("Cannot create probabilities for zero entries");
    }

    std::uniform_real_distribution<Probability> distribution(minInitialWeight,
                                                             maxInitialWeight);
    std::vector<Probability> probabilities;
    probabilities.reserve(count);

    Probability total = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto value = distribution(generator);
        probabilities.push_back(value);
        total += value;
    }

    for (auto& probability : probabilities) {
        probability /= total;
    }

    return probabilities;
}

ProbabilityMatrix randomProbabilityMatrix(std::size_t rows,
                                          std::size_t columns,
                                          std::mt19937& generator) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Cannot create an empty probability matrix");
    }

    ProbabilityMatrix matrix;
    matrix.reserve(rows);

    for (std::size_t row = 0; row < rows; ++row) {
        matrix.push_back(randomProbabilities(columns, generator));
    }

    return matrix;
}

} // namespace

TrainingInput readTrainingInput(std::istream& input, std::ostream& output) {
    output << "Hidden Markov Model training setup\n";
    output << "Use observation indices from 0 to observation_count - 1 in the file.\n";
    output << "Values may be separated by whitespace or commas. Text after # is ignored.\n\n";

    TrainingInput result{};
    result.stateCount = promptPositiveInteger(input,
                                              output,
                                              "Number of states: ",
                                              "Number of states");
    result.observationCount = promptPositiveInteger(input,
                                                    output,
                                                    "Number of observations: ",
                                                    "Number of observations");

    while (true) {
        const auto prompt = "Observation sequence file path (valid indices 0-" +
                            std::to_string(result.observationCount - 1) + "): ";
        const auto filePath = promptRequiredLine(input,
                                                output,
                                                prompt,
                                                "Observation sequence file path");

        try {
            result.observationSequence =
                readObservationSequenceFile(filePath, result.observationCount);
            break;
        } catch (const std::exception& error) {
            output << error.what() << '\n';
        }
    }

    result.maxIterations = promptPositiveIntegerWithDefault(
        input,
        output,
        "Maximum Baum-Welch iterations [100]: ",
        "Maximum Baum-Welch iterations",
        defaultMaxIterations);
    result.tolerance = promptPositiveProbabilityWithDefault(
        input,
        output,
        "Convergence tolerance [1e-6]: ",
        "Convergence tolerance",
        defaultTolerance);

    return result;
}

std::vector<std::size_t> readObservationSequenceFile(
    const std::string& filePath,
    std::size_t observationCount) {
    if (observationCount == 0) {
        throw std::invalid_argument("Observation count must be greater than zero");
    }

    std::ifstream file(filePath);
    if (!file) {
        throw std::runtime_error("Could not open observation sequence file: " + filePath);
    }

    std::vector<std::size_t> sequence;
    std::string line;
    std::size_t lineNumber = 0;

    while (std::getline(file, line)) {
        ++lineNumber;

        const auto commentStart = line.find('#');
        if (commentStart != std::string::npos) {
            line.erase(commentStart);
        }

        for (auto& ch : line) {
            if (ch == ',' || ch == ';') {
                ch = ' ';
            }
        }

        std::istringstream tokens(line);
        std::string token;
        while (tokens >> token) {
            sequence.push_back(parseObservationIndex(token,
                                                     observationCount,
                                                     lineNumber));
        }
    }

    if (sequence.empty()) {
        throw std::invalid_argument(
            "Observation sequence file must contain at least one observation index");
    }

    return sequence;
}

std::vector<std::string> makeIndexedNames(const std::string& prefix,
                                          std::size_t count) {
    std::vector<std::string> names;
    names.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        names.push_back(prefix + std::to_string(index));
    }

    return names;
}

ModelInitialization makeRandomInitialization(std::size_t stateCount,
                                             std::size_t observationCount) {
    if (stateCount == 0) {
        throw std::invalid_argument("State count must be greater than zero");
    }

    if (observationCount == 0) {
        throw std::invalid_argument("Observation count must be greater than zero");
    }

    std::mt19937 generator(defaultInitializationSeed);
    return {
        randomProbabilities(stateCount, generator),
        randomProbabilityMatrix(stateCount, stateCount, generator),
        randomProbabilityMatrix(stateCount, observationCount, generator),
    };
}

} // namespace hmm
