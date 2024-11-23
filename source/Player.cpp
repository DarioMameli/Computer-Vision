//
// Created by Dario Mameli
//

#include "../include/Player.h"

#include <utility>


Player::Player() {
    boundingBox = BoundingBox();
    segment = cv::Mat();
}

Player::Player(BoundingBox boundingBox, cv::Mat segment) {
    this->boundingBox = std::move(boundingBox);
    this->segment = std::move(segment);
}

Player::Player(cv::Rect rectangle, cv::Mat segment) {
    boundingBox = BoundingBox(std::move(rectangle));
    this->segment = std::move(segment);
}

Player::Player(const cv::Mat& segment) {
    // Convert the input segment to the expected format (8-bit single-channel)
    cv::Mat segmentGray;
    cv::cvtColor(segment, segmentGray, cv::COLOR_BGR2GRAY);

    // Find contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(segmentGray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Initialize variables to store the coordinates of the bounding rectangle
    cv::Rect boundingRect;

    // Iterate through all detected contours
    for (const auto& contour : contours) {
        // Find the bounding rectangle for each contour
        cv::Rect currentRect = cv::boundingRect(contour);

        // Update the bounding rectangle if it's larger
        if (currentRect.area() > boundingRect.area()) {
            boundingRect = currentRect;
        }
    }

    // Draw the bounding rectangle on a copy of the original image
    cv::Mat resultImage = segment.clone();
    cv::rectangle(resultImage, boundingRect, cv::Scalar(0, 0, 255), 2); // Draw a red rectangle

    // Display the result
    //cv::imshow("Bounding Rectangle", resultImage);
    //cv::waitKey(0);

    // Delegate construction to the previously defined constructor
    *this = Player(boundingRect, segment);
}

BoundingBox Player::getBoundingBox() const {
    return boundingBox;
}

cv::Mat Player::getSegment() const {
    return segment;
}

cv::Mat Player::setSegment(cv::Mat segment) {
    cv::Mat oldSegment = this->segment;
    this->segment = std::move(segment);
    return oldSegment;
}

BoundingBox Player::setBoundingBox(BoundingBox bbox) {
    BoundingBox oldBB = boundingBox;
    boundingBox = std::move(bbox);
    return oldBB;
}

int Player::setTeamID(int teamID) {
    int oldTeamID = boundingBox.getTeamID();
    boundingBox.setTeamID(teamID);
    return oldTeamID;
}

bool Player::isInPlayingField(const cv::Mat& segmentedPlayingFieldImage) const {
    cv::Rect r = boundingBox.getRectangle();

    checkConsistencyRectangle(segmentedPlayingFieldImage, r);

    int greenCount = 0; // Counter for green pixels
    int totalPixels = r.width * r.height; // Total pixels in the region

    // Iterate through the region defined by r
    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            cv::Vec3b pixel = segmentedPlayingFieldImage.at<cv::Vec3b>(i, j);

            // Check if the pixel is green
            if (pixel[0] == 0 && pixel[1] == 255 && pixel[2] == 0) {
                greenCount++;
            }
        }
    }

    // Check if at least 10% of the pixels are green
    double greenPercentage = (static_cast<double>(greenCount) / totalPixels) * 100.0;
    return greenPercentage >= 10.0;
}










