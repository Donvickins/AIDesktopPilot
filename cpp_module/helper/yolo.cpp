#include "yolo.hpp"

bool loadClassNames(const std::string &path, std::vector<std::string> &class_names_out)
{
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open())
    {
        std::cerr << "Error: Failed to open class names file: " << path << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        class_names_out.push_back(line);
    }
    ifs.close();
    return true;
}

void processFrameWithYOLO(cv::Mat &frame, cv::dnn::Net &net, const std::vector<std::string> &class_names_list)
{
    if (frame.empty() || net.empty())
    {
        return;
    }

    cv::Mat blob;
    // Create a blob from the image
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // Post-processing
    // YOLOv8 output tensor shape is typically [batch_size, num_classes + 4, num_proposals]
    // e.g., [1, 84, 8400] for COCO (80 classes) + 4 box coords.
    cv::Mat detections = outs[0]; // Assuming the first output is the main detection layer

    // The detections Mat has 3 dimensions. For easier access, treat the relevant part as 2D.
    // detection_data points to the [num_channels, num_proposals] part.
    const int num_channels = detections.size[1];  // e.g., 84 (cx, cy, w, h, class_scores...)
    const int num_proposals = detections.size[2]; // e.g., 8400

    cv::Mat detection_matrix = cv::Mat(num_channels, num_proposals, CV_32F, detections.ptr<float>());

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = frame.cols / (float)YOLO_INPUT_WIDTH;
    float y_factor = frame.rows / (float)YOLO_INPUT_HEIGHT;

    for (int i = 0; i < num_proposals; ++i)
    {                                                                                // Iterate over each proposal column
        cv::Mat proposal_scores = detection_matrix.col(i).rowRange(4, num_channels); // Class scores start from 5th row (index 4)
        cv::Point class_id_point;
        double max_score;
        cv::minMaxLoc(proposal_scores, nullptr, &max_score, nullptr, &class_id_point);

        if (max_score > CONFIDENCE_THRESHOLD)
        {
            confidences.push_back((float)max_score);
            class_ids.push_back(class_id_point.y); // class_id_point.y is the index relative to proposal_scores

            // Box coordinates are cx, cy, w, h
            float cx = detection_matrix.at<float>(0, i);
            float cy = detection_matrix.at<float>(1, i);
            float w = detection_matrix.at<float>(2, i);
            float h = detection_matrix.at<float>(3, i);

            int left = static_cast<int>((cx - w / 2) * x_factor);
            int top = static_cast<int>((cy - h / 2) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);

    for (int idx : nms_indices)
    {
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = (class_id < class_names_list.size()) ? class_names_list[class_id] : "Unknown";
        label += cv::format(": %.2f", confidences[idx]);
        cv::putText(frame, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
}