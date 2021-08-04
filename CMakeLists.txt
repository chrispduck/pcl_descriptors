cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MY_GRAND_PROJECT)
find_package(PCL 1.12 REQUIRED)# COMPONENTS common io
include_directories()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set(
    EXAMPLES
    shot_test
    correspondence_grouping
    compute_esf
    compute_shot
    compute_fpfh
    keypoints_extraction
    test_icp
)

foreach(example ${EXAMPLES})
    add_executable(${example}  ${example}.cpp )
    target_link_libraries(${example} ${PCL_LIBRARIES} -lpthread)
endforeach()
