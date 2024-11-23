//
// Created by Dario Mameli
//

#ifndef DETECTION_SEGMENTATION_H_
#define DETECTION_SEGMENTATION_H_

#include "Player.h"
#include <exception>
#include <string>
#include <utility>

#ifndef SPORTVIDEOANALYSIS_SADVISION_PLAYERSCLASSIFIER_H
#define SPORTVIDEOANALYSIS_SADVISION_PLAYERSCLASSIFIER_H

#endif //SPORTVIDEOANALYSIS_SADVISION_PLAYERSCLASSIFIER_H

/**
 * Error class for managing the failure of YOLO to find any segments for a specific image.
 */
class SegmentsNotFoundException : public std::exception {
public:
    /**
     * Constructor of the exception receiving a message to display.
     * @param message the error message.
     */
    explicit SegmentsNotFoundException(std::string message) : errorMessage(std::move(message)) {}

    /**
     * Override the what() method to provide a custom error message.
     * @return the error message
     */
    const char* what() const noexcept override {
        return errorMessage.c_str();
    }

private:
    std::string errorMessage;
};


/**
 * Builds a vector of players for each segment read from the masks predicted by YOLO on a certain image.
 * @param masks_directory_path the path to the directory of the masks.
 * @param image_path the path to the image.
 * @return the vector of players built after the provided segments.
 */
std::vector<Player> extractPlayersFromSegments(const std::string& masks_directory_path, const std::filesystem::path& image_path);
/**
 * Classify the players using k-means (k=2) clustering based on the feature vector computed by calculateColorHistogramFeature
 * on each segment, and save all masks and bounding boxes.
 * @param players the players to be classified.
 * @param image_path the path to the reference image for correct masks name saving.
 * @param masksPath the path to the directory where to save the masks.
 * @param bboxImagesPath the path to the directory where to save the images with the detected bounding boxes.
 */
void classifyPlayers(std::vector<Player>& players, const std::filesystem::path& image_path, const cv::Mat& imagePlayingField_segmented, const std::string& masksPath, const std::string& bboxImagesPath);
/**
 * Merging function for the detection and segmentation of players and playing field. Calls the extractPlayersFromSegments
 * and classifyPlayers functions for player detection and segmentation. Receives the image output of the color features
 * segmentation and clusters the segments into playing field and background.
 * @param masks_directory_path the path to the directory of the masks saved by YOLO segmentation of players.
 * @param image_path the path to the image on which to conclude the detection and segmentation.
 * @param segmented_playingField_image the image subjected to color features segmentation.
 * @param masksPath the path to the directory where to save the masks.
 * @param bboxImagesPath the path to the directory where to save the images with the detected bounding boxes.
 */
void detectionANDsegmentation(const std::string& masks_directory_path, const std::filesystem::path& image_path, const cv::Mat& segmented_playingField_image, const std::string& masksPath, const std::string& bboxImagesPath);

#endif