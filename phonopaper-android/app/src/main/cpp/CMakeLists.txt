cmake_minimum_required(VERSION 3.10.2)

project("phonopaper-android")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Find JNI
find_package(JNI REQUIRED)

# Find log library
find_library(log-lib log)

# Build the Rust library first
# The Rust library will be built via cargo and linked here

# Build the C++ JNI wrapper
add_library(native-lib SHARED
    native-lib.cpp
)

# Link against log and the Rust library
find_library(rust-lib phonopaper_android PATHS ${CMAKE_BINARY_DIR})
target_link_libraries(native-lib 
    ${log-lib} 
    ${rust-lib}
    ${JNI_LIBRARIES}
)

# Set the library name to match Java System.loadLibrary call
set_target_properties(native-lib PROPERTIES SUFFIX ".so")
