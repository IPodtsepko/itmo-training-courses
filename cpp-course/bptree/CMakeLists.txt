cmake_minimum_required(VERSION 3.13)

include(test/Strict.cmake)

set(PROJECT_NAME bptree)
project(${PROJECT_NAME})

# Set up the compiler flags
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inlcude directories
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
include_directories(${COMMON_INCLUDES})

# Source files
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Separate executable: main
list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/main.cpp)

if (SRC_FILES)
    add_library(bptree_lib ${SRC_FILES})
    target_compile_options(bptree_lib PUBLIC ${COMPILE_OPTS})
    target_link_options(bptree_lib PUBLIC ${LINK_OPTS})
    setup_warnings(bptree_lib)
endif()

# Main
add_executable(bptree ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_compile_options(bptree PRIVATE ${COMPILE_OPTS})
target_link_options(bptree PRIVATE ${LINK_OPTS})
if (TARGET bptree_lib)
    target_link_libraries(bptree bptree_lib)
endif()
setup_warnings(bptree)

# google test is a git submodule
add_subdirectory(googletest)

enable_testing()

# test is a git submodule
add_subdirectory(test)

add_test(NAME tests COMMAND runUnitTests)
