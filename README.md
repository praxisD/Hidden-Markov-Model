# Hidden Markov Model

A C++20 implementation of a discrete Hidden Markov Model trained with
the Baum-Welch algorithm.

The project uses only the C++ standard library and CMake.

## Requirements

- CMake 3.20 or newer
- A C++20-capable compiler

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
└── data/
    ├── example_observations.txt
    └── README.md
```

## Build

```bash
cmake -S . -B build
cmake --build build
./build/hidden-markov-model
```

## Usage

The program prompts for:

1. The number of hidden states.
2. The number of possible observations.
3. A text file containing the observation sequence.
4. Optional Baum-Welch iteration and tolerance settings.

For example, after building:

```text
Number of states: 2
Number of observations: 3
Observation sequence file path (valid indices 0-2): data/example_observations.txt
Maximum Baum-Welch iterations [100]:
Convergence tolerance [1e-6]:
```

Observation files use zero-based observation indices separated by whitespace,
commas, or semicolons. Text after `#` is ignored.

```text
0, 1, 2, 1, 0
```

Initial probabilities, transition rows, and emission rows are created
automatically from a deterministic random seed and then normalized, so repeated runs
start from the same parameters.

## Notes

The current implementation performs probability calculations in standard
floating-point space. This keeps the code direct and readable, but very long
observation sequences may require a log-space implementation to avoid numerical
underflow.

## Future Additions

1. Add log-space implementation.
2. Add mixture-model functionality for emission probabilities.
