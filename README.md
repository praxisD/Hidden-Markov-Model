# Hidden Markov Model

A small C++20 project for building a Hidden Markov Model step by step.

## Project Layout

```text
.
├── CMakeLists.txt
├── include/
│   └── hmm/
│       └── HiddenMarkovModel.hpp
├── src/
│   ├── HiddenMarkovModel.cpp
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

## Suggested Steps

1. Define the states and observations.
2. Store start, transition, and emission probabilities.
3. Add sequence probability evaluation.
4. Add the forward algorithm.
5. Add the Viterbi algorithm.
6. Add training later with Baum-Welch.
