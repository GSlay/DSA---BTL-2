cmake_minimum_required(VERSION 3.0)
project(MyProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories for headers
include_directories(
    Assignment2/Code/include
    Assignment2/Code/include/ann
    Assignment2/Code/include/tensor
    Assignment2/Code/include/sformat
    Assignment2/Code/demo
)

# Find all .cpp files in the src directory
file(GLOB SOURCES 
    "Assignment2/Code/src/*.cpp" 
    "Assignment2/Code/src/ann/*.cpp" 
    "Assignment2/Code/src/tensor/*.cpp"
)

# Add executable
add_executable(my_project ${SOURCES})
