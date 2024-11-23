//
// Created by Alberto Dorizza
//

#ifndef SADVISION_PERFORMANCE_MEASUREMENT_H
#define SADVISION_PERFORMANCE_MEASUREMENT_H

#include "BoundingBox.h"
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>


// ********** mean Intersection over Union (mIoU)  **********

 /**
  * Given the mask paths of prediction and ground truth it returns the mIoU as the average of the computed IoU for each
  * class (background, team A, team B and playfield).
  *
  * Remember that the segmentation ground truth mask has the following coding:
  *   -- 0: background;
  *   -- 1: team 1 player;
  *   -- 2: team 2 player;
  *   -- 3: field.
  * @param bin_pred_path path of the predicted binary mask
  * @param bin_ground_truth_path path of the true binary mask
  * @return mIoU, i.e. the average of the computed IoU for each class (background, team A, team B and playfield).
  */
float mIoU(std::string bin_pred_path, std::string bin_ground_truth_path);

/**
 * Given the binary matrix of prediction and ground truth, it returns the IoU
 * @param prediction binary image of prediction
 * @param ground_truth binary image of ground truth
 * @return IoU ( = intersection / union )
 */
float IoU(const cv::Mat& prediction, const cv::Mat& ground_truth);



// ********** mean Average Precision (mAP) **********

/**
 * It calculates the mean Average Precision between the predicted and true bounding boxes.
 * @param image_path The path of the original image
 * @param pred_bbs A vector of BoundingBox that represents the predicted bounding boxes
 * @param true_bbs A vector of BoundingBox that represents the true bounding boxes
 * @return The mean Average Precision
 */
float mAP(std::string image_path, std::vector<BoundingBox> pred_bbs, std::vector<BoundingBox> true_bbs);

/**
 * It calculates the mean Average Precision between the predicted and true bounding boxes.
 * @param image_path The path of the original image
 * @param pred_bb_path The path to the file where the predicted bounding boxes are stored
 * @param true_bb_path The path to the file where the true bounding boxes are stored
 * @return The mean Average Precision
 */
float mAP(std::string image_path, std::string pred_bb_path, std::string true_bb_path);

/**
 * Function to calculate Average Precision (AP) using the 11-Point Interpolation Method
 * @param precisionRecallCurve Vector that contains all Precision-Recall points
 * @return The Average Precision
 */
float calculateAP(const std::vector<cv::Point2f>& precisionRecallCurve);


// ********** UTILS METHODS **********

/**
 * It inverts the labels (teamID field) of all bounding boxes and save those in the relative file if they have a greater mAP than
 * the original ones.
 * @param image_path The path of the original image
 * @param pred_bb_path The path to the file where the predicted bounding boxes are stored
 * @param true_bb_path The path to the file where the true bounding boxes are stored
 * @return true if it stores the inverted labels bounding boxes file, false otherwise
 */
bool adjustTeamIDAccordingToMAP(std::string image_path, std::string pred_bb_path, std::string true_bb_path);


#endif //SADVISION_PERFORMANCE_MEASUREMENT_H
