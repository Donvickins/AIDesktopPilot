add_library(dxdiag dxdiag.cpp dxdiag.hpp)

add_library(yolo yolo.cpp yolo.hpp)

target_include_directories(
    dxdiag PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    yolo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(yolo PRIVATE opencv_world)