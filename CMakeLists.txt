cmake_minimum_required(VERSION 3.20)

project(hidden-markov-model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hidden-markov-model
    src/HiddenMarkovModel.cpp
    src/main.cpp
)

target_include_directories(hidden-markov-model PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
