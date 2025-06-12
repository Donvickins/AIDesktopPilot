cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

project(ai-agent CXX)

add_subdirectory(helper)

add_executable(${PROJECT_NAME} agent.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
target_link_libraries(${PROJECT_NAME} PRIVATE dxdiag yolo d3d11 dxguid)

add_executable(agent_webcam agent_webcam.cpp)
target_include_directories(agent_webcam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
target_link_libraries(agent_webcam PRIVATE yolo)
