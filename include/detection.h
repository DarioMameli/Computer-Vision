//
// Created by Sara Farris
//

#ifndef DETECTION_H_
#define DETECTION_H_

#include "utils.h"
#include "BoundingBox.h"

#ifndef SPORTVIDEOANALYSIS_SADVISION_DETECTION_H
#define SPORTVIDEOANALYSIS_SADVISION_DETECTION_H

#endif //SPORTVIDEOANALYSIS_SADVISION_DETECTION_H

/**
 * Transforms an image into a vector of images each obtained with one of the following transformations:
 * 1) identity transform (no transformation)
 * 2) gaussian blur
 * 3) gaussian blur + laplacian
 * 4) bilateral filter
 * @param src the source image to be processed.
 * @return the array of processed images.
 */
std::vector<cv::Mat> preprocessing(const cv::Mat& src);
/**
 * Predicts bounding boxes at multiple scales via SVM model pretrained on people detection using HOG features.
 * @param imgPreprocessed the image on which to perform the detection.
 * @return the vector of predicted bounding boxes in the form of rectangles.
 */
std::vector<cv::Rect> hogDescriptorSVM(const cv::Mat& imgPreprocessed);
/**
 * Predicts bounding boxes at multiple scales combining the outputs of 3 cascade classifiers trained respectively on lower,
 * upper and full body detection.
 * @param imgPreprocessed the image on which to perform the detection.
 * @return the vector of predicted bounding boxes in the form of rectangles.
 */
std::vector<cv::Rect> haarFeaturesCascadeDetector(const cv::Mat& imgPreprocessed);
/**
 * Detects contours in an image using Canny algorithm, then selects the strongest ones and performs detection on the
 * image with the contours as overlay using pretrained SVM on people detection with HOG features.
 * @param imgPreprocessed the image on which to perform Canny.
 * @return the vector of predicted bounding boxes in the form of rectangles.
 */
std::vector<cv::Rect> contoursHogSVM(const cv::Mat& imgPreprocessed);
/**
 * Algorithm to classify the bounding boxes using k-means (k=2) with as feature vectors the histograms of the color
 * distribution within the bounding boxes.
 * @param image the image from which the bounding boxes come from.
 * @param bboxes the vector of bounding boxes to be classified.
 * @return the image with the classified bounding boxes displayed.
 */
cv::Mat classifyBBoxes(const cv::Mat& image, std::vector<BoundingBox>& bboxes);
/**
 * Performs the detection on each of the preprocessed images of the given vector (its' pointer is given). Uses one of the
 * 3 following algorithms, depending on the given choice:
 * choice=0) hogDescriptorSVM
 * choice=1) haarFeaturesCascadeDetector
 * choice=2) contoursHogSVM
 * Then the predictions for all the preprocessed images are clustered together and saved in the given vector for further
 * classification.
 * @param imagesPreprocessed the pointer to the vector of preprocessed images.
 * @param imageDetection the image to be assigned as the image displaying the detections.
 * @param boxes the vector of bounding boxes on which to save the predictions.
 * @param choice the given choice of algorithm for detection.
 */
void detection(const std::vector<cv::Mat>* imagesPreprocessed, cv::Mat& imageDetection, std::vector<BoundingBox>& boxes, const int& choice);

#endif