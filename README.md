# Hidden Markov Model

A C++20 project without external dependdencies for training a Hidden Markov Model using the Baum-Welch algorithm.

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

## Future Additions and Improvements

1. Add log-space implementation.
2. Add mixture-model functionality for emission probabilities.
