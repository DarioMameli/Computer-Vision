//
// Created by Dario Mameli
//

#ifndef PLAYER_H_
#define PLAYER_H_

#include "detection.h"

#ifndef SPORTVIDEOANALYSIS_SADVISION_SEGMENTATION_H
#define SPORTVIDEOANALYSIS_SADVISION_SEGMENTATION_H

#endif //SPORTVIDEOANALYSIS_SADVISION_SEGMENTATION_H

/**
 * This class encapsulates a bounding box and a segment for better handling of both detection and segmentation.
 */
class Player {
private:
    // His bounding box
    BoundingBox boundingBox;
    // His binary segment
    cv::Mat segment;

public:
    Player();
    /**
     * Builds a Player object with its' associated bounding box and segment.
     * @param boundingBox the bounding box.
     * @param segment the segment.
     */
    explicit Player(BoundingBox boundingBox, cv::Mat segment);
    /**
     * Builds a Player object creating its' bouding box from the rectangle and the segment provided.
     * @param rectangle the rectangle.
     * @param segment the segment.
     */
    explicit Player(cv::Rect rectangle, cv::Mat segment);
    /**
     * Builds a Player object from a given segment associating the best fit bounding box for the segment.
     * @param segment the segment.
     */
    explicit Player(const cv::Mat& segment);

    /// Getter functions
    BoundingBox getBoundingBox() const;
    cv::Mat getSegment() const;

    /// Setter functions
    BoundingBox setBoundingBox(BoundingBox bbox);
    cv::Mat setSegment(cv::Mat segment);
    /**
     * Sets the teamID to the bounding box.
     * @param teamID the teamID to associate.
     * @return the old teamID.
     */
    int setTeamID(int teamID);

    /**
     * Checks whether the player is in the playing field of the specified image, assuming it is indeed a segmented playing
     * field image.
     * @param segmentedPlayingFieldImage the image on which to perform the check
     * @return true if it is in the playing field, false otherwise.
     */
    bool isInPlayingField(const cv::Mat& segmentedPlayingFieldImage) const;
};

#endif