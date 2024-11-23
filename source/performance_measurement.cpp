//
// Created by Alberto Dorizza
//

#include "../include/performance_measurement.h"
#include "../include/utils.h"


// ********** mean Intersection over Union (mIoU)  **********

float mIoU(std::string bin_pred_path, std::string bin_ground_truth_path) {
    // Read the predicted binary mask and the corresponded ground truth
    cv::Mat bin_mask = imread(bin_ground_truth_path, cv::IMREAD_GRAYSCALE);
    cv::Mat pred_mask = imread(bin_pred_path, cv::IMREAD_GRAYSCALE);

    // Utility variables
    std::vector<double> iou_classes(4);
    cv::Mat temp_mask(bin_mask.size(), CV_8U, cv::Scalar(0));
    cv::Mat temp_pred(pred_mask.size(), CV_8U, cv::Scalar(0));

    for (int i = 0; i < iou_classes.size(); ++i) {
        temp_mask = bin_mask.clone();
        temp_pred = pred_mask.clone();

        // Set pixels that belong to other classes to 0
        temp_mask.setTo(255, bin_mask == i);
        temp_pred.setTo(255, pred_mask == i);

        // Compute and save the single IoU of the class i
        iou_classes.at(i) = IoU(temp_pred, temp_mask);
    }

    // Calculate the mean of all computed IoUs
    double sum = 0;
    for (int i = 0; i < iou_classes.size(); ++i)
        sum += iou_classes.at(i);

    return (float)sum / iou_classes.size();
}

float IoU(const cv::Mat& prediction, const cv::Mat& ground_truth) {
    // intersection between two mask
    cv::Mat intersection(prediction.size(), CV_8U, cv::Scalar(0));
    ground_truth.copyTo(intersection, prediction);

    /*
    cv::imshow("prediction", prediction);
    cv::imshow("ground_truth", ground_truth);
    cv::imshow("intersection", intersection);
    cv::waitKey(0);
     */

    // number of pixels of intersection
    int intersectionPixels = countNonZero(intersection);

    // number of pixels of union
    int union_area = countNonZero(prediction) + countNonZero(ground_truth) - intersectionPixels;

    return (float) intersectionPixels / union_area;
}



// ********** mean Average Precision (mAP) **********

float mAP(std::string image_path, std::vector<BoundingBox> pred_bbs, std::vector<BoundingBox> true_bbs) {
    const float IoU_THRESHOLD = 0.5;
    const int num_classes = 2;

    // read the original image
    cv::Mat img = cv::imread(image_path);

    // vector for storing single AP
    std::vector<float> APs;

    int cumulativeTP, cumulativeFP, totalGroudTruths;
    std::vector<cv::Point2f> precisionRecall;

    // for all classes
    for (int i = 1; i <= num_classes ; ++i) {
        cumulativeTP = 0;
        cumulativeFP = 0;

        totalGroudTruths = 0;
        for (BoundingBox bb : true_bbs)
            if(bb.getTeamID()==i)
                totalGroudTruths++;

        // for all predicted bounding boxes
        for (BoundingBox pred_bb : pred_bbs) {
            bool matched = false;

            // Creates a Mat with a white-filled rectangle that identifies the bounding box area
            cv::Mat tempPredBB(img.size(), CV_8U, cv::Scalar(0));
            cv::Rect tempPredRect;
            tempPredRect = pred_bb.getRectangle();
            cv::rectangle(tempPredBB, cv::Point(tempPredRect.x, tempPredRect.y),cv::Point(tempPredRect.x+tempPredRect.width, tempPredRect.y+tempPredRect.height), cv::Scalar(255), -1);

            // if current bb has label i
            if(pred_bb.getTeamID() == i) {

                // iterate over all true bounding boxes
                for (BoundingBox bb: true_bbs) {

                    // if current bb has label i
                    if(bb.getTeamID() == i) {
                        // Creates a Mat with a white-filled rectangle that identifies the true bounding box area
                        cv::Mat tempBB(img.size(), CV_8U, cv::Scalar(0));
                        cv::Rect tempRect;
                        tempRect = bb.getRectangle();
                        cv::rectangle(tempBB, cv::Point(tempRect.x, tempRect.y),cv::Point(tempRect.x+tempRect.width, tempRect.y+tempRect.height), cv::Scalar(255), -1);

                        float IoUmeasured = IoU(tempPredBB, tempBB);
                        // check IoU between the two Mat
                        if(IoUmeasured > IoU_THRESHOLD)
                            matched = true;

                    }
                }

                // Increasing cumulativeTP only if a match with the true bounding boxes was found, increasing cumulativeFP otherwise
                if(matched) cumulativeTP++;
                else cumulativeFP++;

                // Calculate precision and recall
                float precision = (float)cumulativeTP/(cumulativeTP+cumulativeFP);
                float recall = (float)cumulativeTP/totalGroudTruths;

                // add the point to the Precision-Recall curve represented by the vector "precisionRecall"
                precisionRecall.push_back(cv::Point2f(precision, recall));
            }
        }

        // Sort precisionRecall vector by increasing order
        std::sort(precisionRecall.begin(), precisionRecall.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y <  b.y;
        });

        // Calculate AP
        APs.push_back(calculateAP(precisionRecall));

        // Empties the vector
        precisionRecall.clear();
    }

    // Returns the mean of all APs
    float sumAP = 0;
    for(float ap : APs)
        sumAP += ap;

    return (float)1/num_classes * sumAP;
}

float mAP(std::string image_path, std::string pred_bb_path, std::string true_bb_path) {
    // Read predicted and true bounding boxes from files
    std::vector<BoundingBox> im_bbs = readBoundingBoxes(true_bb_path);
    std::vector<BoundingBox> pred_bbs = readBoundingBoxes(pred_bb_path);

    return mAP(image_path, pred_bbs, im_bbs);
}

float calculateAP(const std::vector<cv::Point2f>& precisionRecallCurve) {
    double ap = 0.0;

    // Recall levels for 11-point interpolation
    std::vector<double> recallLevels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    for (double recallThreshold : recallLevels) {

        // Find the maximum precision for recall >= recallThreshold
        double maxPrecision = 0.0;
        for (const cv::Point2f& point : precisionRecallCurve) {
            if (point.y >= recallThreshold) {
                maxPrecision = std::max(maxPrecision, (double)point.x);
            }
        }

        // Add the precision at this recall level to AP
        ap += maxPrecision;
    }

    // Divide by the number of recall levels (11) to compute the average
    return (float)ap / recallLevels.size();;
}



// ********** UTILS METHODS **********

bool adjustTeamIDAccordingToMAP(std::string image_path, std::string pred_bb_path, std::string true_bb_path) {
    // Read predicted and true bounding boxes from files
    std::vector<BoundingBox> pred_bbs = readBoundingBoxes(pred_bb_path);
    std::vector<BoundingBox> im_bbs = readBoundingBoxes(true_bb_path);

    // Creates new vector of bbs by switching the labels of the "pred_bb_path" input bounding boxes
    std::vector<BoundingBox> invertedLabels;
    for(BoundingBox bb : pred_bbs) {
        bb.getTeamID() == 1 ? bb.setTeamID(2) : bb.setTeamID(1);
        invertedLabels.push_back(bb);
    }

    // Checks if the bounding boxes with the inverted labels have greater mAP
    float mapOriginal = mAP(image_path, pred_bbs, im_bbs);
    float mapInverted = mAP(image_path, invertedLabels, im_bbs);
    if(mapInverted > mapOriginal) {
        // Saves bounding boxes with the inverted labels
        saveBoundingBoxes(pred_bb_path, invertedLabels);

        // Computes the path of the bin and color segmentation results
        std::string segmentation_bin_path = pred_bb_path;
        std::string segmentation_color_path = pred_bb_path;
        segmentation_bin_path.replace(segmentation_bin_path.find("_bb.txt"), sizeof("_bb.txt") - 1, "_bin.png");
        segmentation_color_path.replace(segmentation_color_path.find("_bb.txt"), sizeof("_bb.txt") - 1, "_color.png");

        // only if the segmentation files exists
        if (std::filesystem::exists(segmentation_bin_path)) {
            // Read previous bin segmentation result, inverts labels and save the new segmentation
            cv::Mat segmentation_bin = cv::imread(segmentation_bin_path, cv::IMREAD_GRAYSCALE);
            segmentation_bin.setTo(1, segmentation_bin == 2);
            segmentation_bin.setTo(2, segmentation_bin == 1);
            cv::imwrite(segmentation_bin_path, segmentation_bin);
        }

        if (std::filesystem::exists(segmentation_color_path)) {
            // Read previous color segmentation result, inverts labels and save the new segmentation
            cv::Mat segmentation_color = cv::imread(segmentation_color_path);
            cv::Mat result = segmentation_color.clone();

            cv::Mat mask;
            inRange(segmentation_color, cv::Vec3b(255, 0, 0), cv::Vec3b(255, 0, 0), mask);
            result.setTo(cv::Vec3b(0, 0, 255), mask);

            inRange(segmentation_color, cv::Vec3b(0, 0, 255), cv::Vec3b(0, 0, 255), mask);
            result.setTo(cv::Vec3b(255, 0, 0), mask);

            cv::imwrite(segmentation_color_path, result);
        }

        return true;
    }

    return false;
}






