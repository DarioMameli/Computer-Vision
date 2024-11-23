//
// Created by Dario Mameli
//

#ifndef UTILS_H_
#define UTILS_H_

#include "BoundingBox.h"

#ifndef SPORTVIDEOANALYSIS_SADVISION_UTILS_H
#define SPORTVIDEOANALYSIS_SADVISION_UTILS_H

#endif //SPORTVIDEOANALYSIS_SADVISION_UTILS_H

/**
 * Checks that the rectangles coordinates do not exceed the margins of the image. Adjusts the coordinates if needed.
 * @param image the image where to perform the checks.
 * @param rectangles the vector of rectangles whose coordinates need to be checked and adjusted if necessary.
 */
void checkConsistencyRectangles(const cv::Mat& image, std::vector<cv::Rect>& rectangles);
/**
 * Checks that the rectangle coordinates do not exceed the margins of the image. Adjusts the coordinates if needed.
 * @param image the image where to perform the checks.
 * @param r the rectangle whose coordinates need to be checked and adjusted if necessary.
 */
void checkConsistencyRectangle(const cv::Mat& image, cv::Rect& r);
/**
 * Builds a vector of bounding boxes based on the given rectangles.
 * @param rectangles the vector of rectangles to be transformed.
 * @return the vector of corresponding bounding boxes.
 */
std::vector<BoundingBox> getBoundingBoxes(const std::vector<cv::Rect>& rectangles);
/**
 * Extracts a vector of rectangles encapsulated into the given bounding boxes.
 * @param bboxes the vector of bounding boxes.
 * @return the vector of rectangles encapsulated.
 */
std::vector<cv::Rect> getRectangles(const std::vector<BoundingBox>& bboxes);
/**
 * Reads the bounding boxes of an image in the format x y w h teamID specified and saves all in the returning vector
 * @param filePath Path of the file in which reads all bounding boxes
 * @return A vector of BoundingBox that contains all bounding boxes readed in the file stored in the @param filePath
 */
std::vector<BoundingBox> readBoundingBoxes(const std::string& filePath);
/**
 * Saves the coordinates of the given bounding boxes in a txt file at the given path.
 * @param txtPath the path where to save the coordinates.
 * @param bboxes the bounding boxes whose coordinates are saved.
 */
void saveBoundingBoxes(const std::string& txtPath, const std::vector<BoundingBox>& bboxes);
/**
 * Calculates the 1D feature vector representing the flattened histogram of the color distribution of a given image.
 * @param roi the image on which to compute the feature vector.
 * @param numBinsPerChannel the number of bins per channel to consider. 32 is the default number.
 * @return the feature vector.
 */
cv::Mat calculateColorHistogramFeature(const cv::Mat& roi, int numBinsPerChannel = 32);
/**
 * K-means segmentation (K=2) of the playing field and background on the image provided.
 * @param image the image to be segmented.
 * @return the segmented image
 */
cv::Mat kmeansSegmentationPlayingField(const cv::Mat& image);
/**
 * Utility function to save the bounding boxes with the class provided as rectangles on the provided image in the specified
 * folder path.
 * @param image_path the image on which to paint the bounding boxes.
 * @param outputFolderPath the path to the folder in which to save.
 * @param bboxes the bounding boxes.
 */
void saveImageDetections(const std::filesystem::path& image_path, const std::string& outputFolderPath, const std::vector<BoundingBox>& bboxes);

#endif