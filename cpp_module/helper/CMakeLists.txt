set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Use vcpkg installed OpenCV
set(OpenCV_STATIC ON)
find_package(OpenCV CONFIG REQUIRED)

add_library(dxdiag STATIC dxdiag.cpp)
add_library(utils STATIC utils.cpp)
add_library(yolo STATIC yolo.cpp)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please run 'vcpkg install --triplet x64-windows-static' first.")
else()
    message(STATUS "OpenCV found: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
endif()

target_include_directories(
    dxdiag PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(
    yolo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo PUBLIC ${OpenCV_LIBS})
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})