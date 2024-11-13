# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Define the project name and version
project(MyProject VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source files (you can list all .cpp files or use GLOB)
file(GLOB SOURCES "src/*.cpp")

# Add the executable target
add_executable(my_project ${SOURCES})

# Link any required libraries here (e.g., pthread, Boost)
# target_link_libraries(my_project pthread)