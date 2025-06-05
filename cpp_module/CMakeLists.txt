cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(ai-agent)

set(BUILD_SHARED_LIBS OFF)
set(BUILD_opencv_world ON)
# 3. Disable parts of OpenCV you don't need to speed up the build.
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_EXAMPLES OFF)

set(DEP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies)

add_subdirectory(${DEP_DIR}/opencv)

add_executable(${PROJECT_NAME} agent.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
target_link_libraries(${PROJECT_NAME} PRIVATE dxdiag yolo d3d11 dxguid)

add_executable(agent_webcam agent_webcam.cpp)
target_include_directories(agent_webcam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
target_link_libraries(agent_webcam PRIVATE yolo)
