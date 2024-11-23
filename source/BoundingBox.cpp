//
// Created by Dario Mameli
//

#include "../include/BoundingBox.h"

BoundingBox::BoundingBox() {
    rect = cv::Rect();
}

BoundingBox::BoundingBox(cv::Rect rect) {
    this->rect = std::move(rect);
}

BoundingBox::BoundingBox(cv::Rect rect, int teamID) {
    this->rect = std::move(rect);
    this->teamID = teamID;
}

cv::Rect BoundingBox::getRectangle() const {
    return rect;
}

int BoundingBox::getTeamID() const {
    return teamID;
}

cv::Rect BoundingBox::setRectangle(cv::Rect rect) {
    cv::Rect oldRect = std::move(this->rect);
    this->rect = std::move(rect);
    return oldRect;
}

int BoundingBox::setTeamID(int teamID) {
    int oldTeamID = this->teamID;
    this->teamID = teamID;
    return oldTeamID;
}






