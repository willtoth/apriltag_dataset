cmake_minimum_required(VERSION 3.16)
project(apriltag_bench LANGUAGES C)

set(CMAKE_C_STANDARD 99)

# Default to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -ftree-vectorize")

# Apriltag library location
set(APRILTAG_DIR "/home/will/Src/apriltag" CACHE PATH "Path to apriltag source")
set(APRILTAG_BUILD_DIR "${APRILTAG_DIR}/build" CACHE PATH "Path to apriltag build")

add_executable(bench_apriltag
    bench_apriltag.c
    cJSON.c
)

target_include_directories(bench_apriltag PRIVATE
    ${APRILTAG_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(bench_apriltag PRIVATE ${APRILTAG_BUILD_DIR})
target_link_libraries(bench_apriltag apriltag m pthread)

set_target_properties(bench_apriltag PROPERTIES
    BUILD_RPATH "${APRILTAG_BUILD_DIR}"
    INSTALL_RPATH "${APRILTAG_BUILD_DIR}"
)
