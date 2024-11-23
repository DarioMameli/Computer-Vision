//
// Created by Dario Mameli
//

#ifndef BOUNDINGBOX_H_
#define BOUNDINGBOX_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/utility.hpp>
#include <utility>

#ifndef SPORTVIDEOANALYSIS_SADVISION_BOUNDINGBOX_H
#define SPORTVIDEOANALYSIS_SADVISION_BOUNDINGBOX_H

#endif //SPORTVIDEOANALYSIS_SADVISION_BOUNDINGBOX_H

/**
 * This class encapsulates a rectangle representing the coordinates of the bounding box in the format x y w h with the
 * addition of an int value teamID representing the class of the bounding box.
 */
class BoundingBox {
private:
    cv::Rect rect;
    int teamID = 0;

public:
    BoundingBox();
    /**
     *
     * @param rect the rectangle representing the coordinates of the bounding box.
     */
    explicit BoundingBox(cv::Rect rect);
    /**
     *
     * @param rect the rectangle representing the coordinates of the bounding box.
     * @param teamID the team ID. When classification is done it should be either 1 or 2.
     */
    BoundingBox(cv::Rect rect, int teamID);

    /// Getter functions
    cv::Rect getRectangle() const;
    int getTeamID() const;

    /// Setter functions
    cv::Rect setRectangle(cv::Rect rect);
    int setTeamID(int teamID);
};

#endif