# Hidden Markov Model

A small C++20 project for building a Hidden Markov Model step by step.

## Project Layout

```text
.
├── CMakeLists.txt
├── include/
│   └── hmm/
│       ├── HiddenMarkovModel.hpp
│       └── Inputs.hpp
├── src/
│   ├── HiddenMarkovModel.cpp
│   ├── Inputs.cpp
│   └── main.cpp
├── tests/
│   └── README.md
└── data/
    └── README.md
```

## Build

```bash
cmake -S . -B build
cmake --build build
./build/hidden-markov-model
```

The program asks for the number of hidden states, the number of observations,
and a text file containing the observation sequence. Observation files use
zero-based observation indices, separated by whitespace or commas:

```text
0, 1, 2, 1, 0
```

See `data/example_observations.txt` for a small example.

Initial probabilities, transition rows, and emission rows are created
automatically with deterministic mild random values, then normalized.

## Suggested Steps

1. Define the states and observations.
2. Store start, transition, and emission probabilities.
3. Add sequence probability evaluation.
4. Add the forward algorithm.
5. Add training with Baum-Welch.
6. Add the Viterbi algorithm later.
