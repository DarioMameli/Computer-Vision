//
// Created by Dario Mameli
//

#include "../include/utils.h"

std::vector<BoundingBox> getBoundingBoxes(const std::vector<cv::Rect>& rectangles) {
    std::vector<BoundingBox> bboxes {};
    for(const auto& rect : rectangles) {
        BoundingBox bbox(rect);
        bboxes.push_back(bbox);
    }
    return bboxes;
}

std::vector<cv::Rect> getRectangles(const std::vector<BoundingBox>& bboxes) {
    std::vector<cv::Rect> rectangles{};
    for(const BoundingBox& bbox : bboxes) {
        rectangles.push_back(bbox.getRectangle());
    }
    return rectangles;
}

void checkConsistencyRectangle(const cv::Mat& image, cv::Rect& r) {
    // Check width consistency
    if(r.width > image.cols) r.width = image.cols;
    else if(r.width < 0) r.width = 0;

    // Check height consistency
    if(r.height > image.rows) r.height = image.rows;
    else if(r.height < 0) r.height = 0;

    // Check x consistency
    if(r.x < 0) r.x = 0;
    else if (r.x > image.cols) r.x = image.cols;

    // Check y consistency
    if(r.y < 0) r.y = 0;
    else if (r.y > image.rows) r.y = image.rows;

    // Check x+w consistency
    if(r.x + r.width > image.cols) r.width = image.cols - r.x;

    // Check y+h consistency
    if(r.y + r.height > image.rows) r.height = image.rows - r.y;
}

void checkConsistencyRectangles(const cv::Mat& image, std::vector<cv::Rect>& rectangles) {
    for(cv::Rect& r : rectangles) {
        checkConsistencyRectangle(image, r);
    }
}

std::vector<BoundingBox> readBoundingBoxes(const std::string& filePath) {
    std::vector<BoundingBox> predictions;
    std::ifstream inputFile(filePath);

    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file: " << filePath << std::endl;
        return predictions; // Return an empty vector
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        BoundingBox prediction{};
        double x, y, w, h, teamID;
        iss >> x >> y >> w >> h >> teamID;
        int x_r, y_r, w_r, h_r, h_teamID;
        x_r = static_cast<int>(x);
        y_r = static_cast<int>(y);
        w_r = static_cast<int>(w);
        h_r = static_cast<int>(h);
        h_teamID = static_cast<int>(teamID);

        prediction.setRectangle(cv::Rect(x_r, y_r, w_r, h_r));
        prediction.setTeamID(h_teamID);
        predictions.push_back(prediction);
    }

    inputFile.close();

    return predictions;
}

void saveBoundingBoxes(const std::string& txtPath, const std::vector<BoundingBox>& bboxes) {
    std::ofstream outputFile(txtPath);

    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file: " << txtPath << std::endl;
        return;
    }

    for (const BoundingBox& bbox : bboxes) {
        outputFile << bbox.getRectangle().x << " " << bbox.getRectangle().y << " " << bbox.getRectangle().width << " " << bbox.getRectangle().height << " " << bbox.getTeamID() << std::endl;
    }

    outputFile.close();

    std::cout << "Bounding boxes saved to " << txtPath << std::endl;
}

void saveImageDetections(const std::filesystem::path& image_path, const std::string& outputFolderPath, const std::vector<BoundingBox>& bboxes) {
    cv::Mat image = cv::imread(image_path.string());
    std::string imageFilename = image_path.filename().string();
    std::string imageName = imageFilename.substr(0, imageFilename.length()-4);
    std::string outputFilePath = outputFolderPath+imageName+"_detection.jpg";
    cv::Mat overlay = image.clone();
    // Loop through the rectangles and draw them on the image
    for (const BoundingBox& bbox : bboxes) {
        if(bbox.getTeamID()==1) {
            cv::rectangle(overlay, bbox.getRectangle(), cv::Scalar(255, 0, 0), -1); // Blue area
            cv::rectangle(image, bbox.getRectangle(), cv::Scalar(255, 0, 0), 2); // Blue border
        }
        else {
            cv::rectangle(overlay, bbox.getRectangle(), cv::Scalar(0, 0, 255), -1); // Red area
            cv::rectangle(image, bbox.getRectangle(), cv::Scalar(0, 0, 255), 2); // Red border
        }
    }
    cv::addWeighted(image, 0.5, overlay, 0.5, 0, image);
    cv::imwrite(outputFilePath, image);
    std::cout << "Saved " << outputFilePath << std::endl;
}

cv::Mat calculateColorHistogramFeature(const cv::Mat& roi, int numBinsPerChannel) {
    // Convert ROI to LAB color space for better color representation
    cv::Mat labRoi;
    cv::cvtColor(roi, labRoi, cv::COLOR_BGR2Lab);

    // Split LAB channels
    std::vector<cv::Mat> labChannels;
    cv::split(labRoi, labChannels);

    // Define histogram structure
    int histSize[] = {numBinsPerChannel, numBinsPerChannel, numBinsPerChannel};
    float range[] = {1, 256};  // L, A, and B channels have a range of [1, 255]. Exclude black color 0
    const float* ranges[] = {range, range, range};
    int channels[] = {0, 1, 2};  // L, A, B channels

    // Compute the histogram
    cv::Mat histogram;
    cv::calcHist(&labChannels[0], 3, channels, cv::Mat(), histogram, 3, histSize, ranges);

    // Normalize the histogram
    histogram /= cv::sum(histogram)[0];

    return histogram.reshape(1, 1);  // Flatten the histogram into a 1D feature vector
}

cv::Mat kmeansSegmentationPlayingField(const cv::Mat& image) {

    // Reshape the image to a 2D matrix of pixels (rows x columns) x 3 (RGB)
    cv::Mat reshaped_image = image.reshape(1, image.rows * image.cols);
    reshaped_image.convertTo(reshaped_image, CV_32F);

    // 2 classes: background and playing field
    int K = 2;

    // Perform k-means clustering
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2);
    cv::Mat labels, centers;
    cv::kmeans(reshaped_image, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);

    // Determine which cluster is mostly located below (playingField) based on a cost function
    double max_normalized_cost = -1.0;
    int playingField = -1;

    // For each cluster
    for (int label = 0; label < K; label++) {
        // Initialization
        double total_squared_distance = 0.0;
        int cluster_size = 0;
        // Scan the image
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int pixel_label = labels.at<int>(i * image.cols + j, 0);
                if (pixel_label == label) {
                    // Calculate the squared distance of the pixel along the y-axis from 0
                    double squared_distance = static_cast<double>(i) * static_cast<double>(i);
                    // Calculate the sum of squared distances
                    total_squared_distance += squared_distance;
                    cluster_size++;
                }
            }
        }

        // Calculate the normalized cost by dividing the sum of squared distances by the cluster size
        double normalized_cost = (cluster_size > 0) ? (total_squared_distance / static_cast<double>(cluster_size)) : 0.0;
        // The playing field will be the cluster with the maximum cost
        if (normalized_cost > max_normalized_cost) {
            max_normalized_cost = normalized_cost;
            playingField = label;
        }
    }


    // Convert labels to CV_32S for indexing
    labels.convertTo(labels, CV_32S);

    // Assign colors to the image based on the cluster positions
    cv::Scalar green(0, 255, 0);  // Green color
    cv::Scalar black(0, 0, 0);    // Black color

    // Create green color to assign to segmented pixels
    cv::Vec3b colorGreen;
    colorGreen[0] = green[0];
    colorGreen[1] = green[1];
    colorGreen[2] = green[2];

    // Create black color to assign to segmented pixels
    cv::Vec3b colorBlack;
    colorBlack[0] = black[0];
    colorBlack[1] = black[1];
    colorBlack[2] = black[2];

    cv::Mat segmented_image(image.size(), image.type());
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int label = labels.at<int>(i * image.cols + j, 0);
            if (label == playingField) {
                segmented_image.at<cv::Vec3b>(i, j) = colorGreen;
            } else {
                segmented_image.at<cv::Vec3b>(i, j) = colorBlack;
            }
        }
    }

    // Display the original and segmented images
    //cv::imshow("CFS Image", image);
    //cv::imshow("Segmented KM Image", segmented_image);
    //cv::waitKey(0);

    return segmented_image;
}