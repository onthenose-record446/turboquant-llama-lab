cmake_minimum_required(VERSION 3.20)

project(turboquant_llama_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(turboquant_core
    cpp/src/turboquant.cpp
)

target_include_directories(turboquant_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

add_executable(turboquant-bench
    cpp/tools/turboquant-bench.cpp
)

target_link_libraries(turboquant-bench
    PRIVATE
        turboquant_core
)
