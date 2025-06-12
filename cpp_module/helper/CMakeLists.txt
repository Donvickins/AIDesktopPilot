set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
add_library(dxdiag STATIC dxdiag.cpp)

add_library(yolo STATIC yolo.cpp)

set(DEP_DIR ${CMAKE_SOURCE_DIR}/dependencies)
set(OpenCV_DIR ${DEP_DIR}/opencv)
set(OpenCV_STATIC ON)

find_package(OpenCV CONFIG REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please ensure it is installed and the path is set correctly.")
else()
    message(STATUS "OpenCV found: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
endif()

target_include_directories(
    dxdiag PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    yolo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo PUBLIC ${OpenCV_LIBS})