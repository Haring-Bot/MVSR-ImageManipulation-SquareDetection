cmake_minimum_required(VERSION 3.0)
project(OpenCV_Test)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable target
add_executable(main src/attempt2.cpp)

# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})
