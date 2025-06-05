#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <fstream>
#include <filesystem>

const float CONFIDENCE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.4f;
const int YOLO_INPUT_WIDTH = 640;
const int YOLO_INPUT_HEIGHT = 640;

void processFrameWithYOLO(cv::Mat &frame, cv::dnn::Net &net, const std::vector<std::string> &class_names_list);
bool loadClassNames(const std::string &path, std::vector<std::string> &class_names_out);