//
// Created by Sara Farris
//

#include "../include/detection.h"

// ************* PRE PROCESSING *****************//

std::vector<cv::Mat> preprocessing(const cv::Mat& src) {

    /// Initial data needed
    std::vector<cv::Mat> images;
    cv::Mat imgGray,imgBlur,imgLaplacian,imgSharpened,imgBilateral;

    // We convert the image into grayscale
    cv::cvtColor(src, imgGray, cv::COLOR_BGR2GRAY);

    std::cout << "--> Pre processing 0: nothing" << std::endl;
    ///Pre processing 0: nothing
    images.push_back(src.clone());

    std::cout << "--> Pre processing 1: gaussian blur" << std::endl;
    ///Pre processing 1: gaussian blur to reduce noise
    cv::GaussianBlur(imgGray, imgBlur, cv::Size(5, 5), 0);
    images.push_back(imgBlur);
/*
        cv::imshow("blur", imgBlur);
        cv::waitKey(0);
*/

    ///Pre processing 2: gaussian blur + laplacian
    std::cout << "--> Pre processing 2: gaussian blur + laplacian" << std::endl;
    Laplacian(imgBlur, imgLaplacian, CV_8U);
    cv::addWeighted(imgBlur, 1.0, imgLaplacian, -1, 0, imgSharpened);
    images.push_back(imgSharpened);
/*
        cv::imshow("blur", imgBlur);
        cv::imshow("laplacianBlur", imgLaplacian);
        cv::imshow("sharpenedBlur", imgSharpened);
        cv::waitKey(0);
*/

    ///Pre processing 3: bilateral filter to reduce noise but keep the edges
    std::cout << "--> Pre processing 3: bilateral filter" << std::endl;
    cv::bilateralFilter(src, imgBilateral, 5, 150, 150);
    images.push_back(imgBilateral);
/*
        cv::imshow("bilateral", imgBilateral);
        cv::waitKey(0);
*/

    return images;

}

//***************** METHODS USING OPENCV *******************//


// First methode: HOG+SVM 
std::vector<cv::Rect> hogDescriptorSVM(const cv::Mat& imgPreprocessed) {
    std::cout << "** Method 0: Hog + SVM **" << std::endl;
    cv::HOGDescriptor hog;
    hog.setSVMDetector(
            cv::HOGDescriptor::getDefaultPeopleDetector());  // the default detector is trained with a dataset of people
            // walking in the street

    std::vector<cv::Rect> found;
    std::vector<double> weights;

    hog.detectMultiScale(imgPreprocessed, found, weights, 0, cv::Size(2, 2), cv::Size(32, 32), 1.05, 0); // detectMultiscale
    // is used to detect people in the image even if in different scale  (Window used by HOG is set to detect people in 64x128 size)

    hog.groupRectangles(found, weights, 5, 0.2);   // it groups close rectangles together

    checkConsistencyRectangles(imgPreprocessed, found);

    /* SHOW FUNCTIONS */
    cv::Mat imageDetection = imgPreprocessed.clone();

    /// draw rectangles in the original image
    for (int i = 0; i < found.size(); i++) {
        cv::Rect r = found[i];
        rectangle(imageDetection, found[i], cv::Scalar(0, 0, 255), 3);
        std::stringstream temp;
        temp << weights[i];
        putText(imageDetection, temp.str(), cv::Point(found[i].x, found[i].y + 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255));
    }
    /// Show
    //cv::imshow("HOG + SVM", imageDetection);
    //cv::waitKey(0);


    return found;
}

//Second method: HAAR FEATURES
std::vector<cv::Rect> haarFeaturesCascadeDetector(const cv::Mat& imgPreprocessed) {
    std::cout << "** Method 1: Haar Features Detector **" << std::endl;

    /// Initial data
    std::vector<cv::Rect> found;
    std::vector<cv::Rect> found1;
    std::vector<cv::Rect> found2;
    std::vector<cv::Rect> found3;

    cv::CascadeClassifier full_body;
    cv::CascadeClassifier upper_body;
    cv::CascadeClassifier lower_body;
    cv::CascadeClassifier hog_body;

    /// haar features of full, upper and lower body
    std::string cascade_full = "../cascade_classifier/haarcascade_fullbody.xml";
    std::string cascade_up = "../cascade_classifier/haarcascade_upperbody.xml";
    std::string cascade_low = "../cascade_classifier/haarcascade_lowerbody.xml";

    // We check whether they are properly loaded
    bool loaded_full = full_body.load(cascade_full);
    bool loaded_up = upper_body.load(cascade_up);
    bool loaded_low = lower_body.load(cascade_low);
    //bool loaded_hog = hog_body.load(cascade_hog); // doesn't work

    /*
    std::cout << "Value:" << std::endl << loaded_full << std::endl << loaded_up << std::endl << loaded_low
              << std::endl //<< loaded_hog << std::endl;
    */

    cv::Mat imageDetection = imgPreprocessed.clone();
    // using detectMultiscale again to check on different size
    if (loaded_full == 1) {
        full_body.detectMultiScale(imgPreprocessed, found, 1.04, 2, 0 | 1, cv::Size(40, 70),
                                    cv::Size(80, 300));
        //std::cout << found.size() << std::endl;

        checkConsistencyRectangles(imgPreprocessed, found);

        /* SHOW
        imageDetection = imgPreprocessed.clone();
        if (found.size() >= 0) {
            for (int i = 0; i < found.size(); i++) {
                rectangle(imageDetection, found[i].tl(), found[i].br(), cv::Scalar(0, 0, 255), 2, 8, 0);
            }

            cv::imshow("detected person full", imageDetection);
            cv::waitKey(0);
        }
         */
    } else std::cout << "ERROR" << std::endl;

    if (loaded_up == 1) {
        upper_body.detectMultiScale(imgPreprocessed, found1, 1.04, 2, 0 | 1, cv::Size(40, 70),
                                    cv::Size(80, 300));
        //std::cout << found1.size() << std::endl;

        checkConsistencyRectangles(imgPreprocessed, found1);

        /* SHOW
        imageDetection = imgPreprocessed.clone();
        if (found1.size() >= 0) {
            for (int i = 0; i < found1.size(); i++) {
                rectangle(imageDetection, found1[i].tl(), found1[i].br(), cv::Scalar(0, 0, 255), 2, 8, 0);
            }

            cv::imshow("detected person up", imageDetection);
            cv::waitKey(0);
        }
         */
    } else std::cout << "ERROR" << std::endl;

    if (loaded_low == 1) {
        lower_body.detectMultiScale(imgPreprocessed, found2, 1.04, 1, 0, cv::Size(10, 10), cv::Size(200, 200));
        //std::cout << found2.size() << std::endl;

        checkConsistencyRectangles(imgPreprocessed, found2);

        /* SHOW
        imageDetection = imgPreprocessed.clone();
        if (found2.size() >= 0) {
            for (int i = 0; i < found2.size(); i++) {
                rectangle(imageDetection, found2[i].tl(), found2[i].br(), cv::Scalar(0, 0, 255), 2, 8, 0);
            }
            cv::imshow("detected person low", imageDetection);
            cv::waitKey(0);
        }
         */
    } else std::cout << "ERROR" << std::endl;

    // DOESN'T WORK (pedestrian haar features)
    /*if (loaded_hog == 1) {
        lower_body.detectMultiScale(imgPreprocessed, found3, 1.04, 1, 0, cv::Size(10, 10), cv::Size(200, 200));
        std::cout << found3.size() << std::endl;

         checkConsistencyRectangles(imgPreprocessed, found3);

        if (found3.size() >= 0) {
            for (int i = 0; i< found3.size();i++) {
                rectangle(imgOriginal, found3[i].tl(), found3[i].br(), cv::Scalar(0, 0, 255), 2, 8, 0);
            }

            cv::imshow("detected person low", imgOriginal);
            cv::waitKey(0);
        }
    } else std::cout << "ERROR" << std::endl;
    */

    std::vector<cv::Rect> rectangles;

    // Merge all predictions
    for(const auto& i : found) rectangles.push_back(i);
    for(const auto& i : found1) rectangles.push_back(i);
    for(const auto& i : found2) rectangles.push_back(i);
    for(const auto& i : found3) rectangles.push_back(i);

    /*
    // Loop through the rectangles and draw bounding boxes on the image
    for (const cv::Rect& rect : rectangles)
        cv::rectangle(imageDetection, rect, cv::Scalar(0, 0, 255), 2);
    cv::imshow("imageDetection", imageDetection);
    cv::waitKey(0);
    */

    checkConsistencyRectangles(imgPreprocessed, rectangles);

    return rectangles;
}

// Third method: CONTOUR+HOG+SVM
std::vector<cv::Rect> contoursHogSVM(const cv::Mat& imgPreprocessed) {
    std::cout << "** Method 2: Contours, HOG and SVM **" << std::endl;

    /// Initial data
    cv::Mat canny, eroded, dilated, final_image, final_dilate;
    cv::Mat imgContours = imgPreprocessed.clone();

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    /// Pre-processing of the image
    cv::Canny(imgPreprocessed, canny, 100, 80);

    ///Using morphological operators doesn't give us more defined lines, in some cases the output gets worst
    // tried different combinations of them
    //cv::dilate(canny,dilated, cv::Mat(), cv::Point(-1,1),1);
    //cv::erode(canny,eroded, cv::Mat(), cv::Point(-1,1),1);
    //cv::imshow("Canny", canny);
    //cv::waitKey(0);

    //cv::erode(dilated,eroded, cv::Mat(), cv::Point(-1,1),1);
    //cv::dilate(eroded,final_dilate, cv::Mat(), cv::Point(-1,1),1);


    ///Find contours from Canny lines
    findContours(canny, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    int idx = 0, largestComp = 0;
    double minArea = 0;
    std::map<int, int> largestContours;

    /// We select the contours that are stronger from each hierarchy level, so as to use them as base for detection
    for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
        const std::vector<cv::Point> &c = contours[idx];
        double area = fabs(contourArea(cv::Mat(c)));
        if (area > minArea) {
            largestContours[hierarchy[idx][0]] = idx;
        }
    }
    //cv::Scalar color(0, 0, 255);
    /// Draw contours
    //cv::drawContours(imgOriginal, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);

    for (const auto &entry: largestContours) {
        int hierarchyLevel = entry.first;
        int contourIdx = entry.second;

        // Retrieve the contour using the contour index
        const std::vector<cv::Point> &c = contours[contourIdx];

        // Draw the contour on the output image
        cv::Scalar color(0, 0, 255); // Red color for drawing the contours
        drawContours(imgContours, std::vector<std::vector<cv::Point>>{c}, 0, color, 2); // Draw the contour
    }

    //cv::imshow("Contours", imgContours);
    //cv::waitKey(0);

    /*
    /// Approximate contours to polygons, creating rectangles from contour (doesn't work because there are too many "noisy contours")
    cv::RNG rng(12345);
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<cv::Point2f>center( contours.size() );
    std::vector<float>radius( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
    { approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
        minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
    }


    /// Draw polygonal contour + bonding rects + circles
    cv::Mat drawing = cv::Mat::zeros( imgPreprocessed.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours_poly, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
    }

   */

    /// Creating rectangles using HOG as in the other cases
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    std::vector<cv::Rect> found;
    std::vector<double> weights;

    hog.detectMultiScale(imgContours, found, weights, 0, cv::Size(2, 2), cv::Size(32, 32), 1.059, 0);

    hog.groupRectangles(found, weights, 5, 0.2);

    checkConsistencyRectangles(imgContours, found);

    /* SHOW
    cv::Mat imageDetection = imgPreprocessed.clone();
    for (int i = 0; i < found.size(); i++) {
        if (weights[i] > 0.5) {
            cv::Rect r = found[i];
            rectangle(imageDetection, found[i], cv::Scalar(0, 0, 255), 3);
            std::stringstream temp;
            temp << weights[i];
            putText(imageDetection, temp.str(), cv::Point(found[i].x, found[i].y + 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 255));
        }
    }

    cv::imshow("Detection method 3", imageDetection);
    cv::waitKey(0);
     */

    return found;
}

//**********************MANAGING OPTIONS****************//

void detection(const std::vector<cv::Mat>* imagesPreprocessed, cv::Mat& imageDetection, std::vector<BoundingBox>& bboxes, const int& choice) {

    // ********** LOCALIZATION ******************//

    std::vector<cv::Rect> rectangles; // temporary vector of predictions
    std::vector<cv::Rect> allRectangles; // vector of all predictions

    for(const auto& imgPreprocessed : *imagesPreprocessed) {

        if (choice == 0) {  /// Using just HOG descriptor, calculates features based on gradient, pass them to an SVM classifier
            rectangles = hogDescriptorSVM(imgPreprocessed);
        }
        else if (choice == 1) {  /// Using haar classifiers of full, lower, upper body taken from Opencv
            rectangles = haarFeaturesCascadeDetector(imgPreprocessed);
        }
        else if (choice == 2) {  /// Using contours and HOG descriptor, then pass the features to SVM classifier
            rectangles = contoursHogSVM(imgPreprocessed);
        }

        for(const auto& r : rectangles)
            allRectangles.emplace_back(r);

    }

    cv::Mat imageLocalization_all = imageDetection.clone();

    // Loop through the rectangles and draw bounding boxes on the image
    for (const cv::Rect& rect : allRectangles)
        cv::rectangle(imageLocalization_all, rect, cv::Scalar(0, 0, 255), 2);

    //cv::imshow("All Localization", imageLocalization_all);
    //cv::waitKey(0);

    // Cluster predictions
    cv::groupRectangles(allRectangles, 2);

    cv::Mat imageLocalization = imageDetection.clone();

    for (cv::Rect & r : allRectangles)
        rectangle(imageLocalization, r, cv::Scalar(0, 0, 255), 2);

    //cv::imshow("Clustered Localization", imageLocalization);
    //cv::waitKey(0);

    // ********** CLASSIFICATION ******************//

    checkConsistencyRectangles(imageDetection, allRectangles);

    //we retrive the rectangles found which are our bboxes
    bboxes = getBoundingBoxes(allRectangles);

    // Abort if no bounding boxes are found/left
    if(allRectangles.empty())
        return;

    imageDetection = classifyBBoxes(imageDetection, bboxes);

    //cv::imshow("Detection", imageDetection);
    //cv::waitKey(0);

}

cv::Mat classifyBBoxes(const cv::Mat& image, std::vector<BoundingBox>& bboxes) {
    // Define the number of classes (2 in this case)
    int numClasses = 2;

    // Create an overlay image for visualization with transparency
    cv::Mat overlay = image.clone();

    // Mapping teamID number to color
    std::map<int, cv::Scalar> colorMap = {
            {1, cv::Scalar(255, 0, 0)}, // Blue -> Team 1
            {2, cv::Scalar(0, 0, 255)} // Red -> Team 2
    };

    std::vector<cv::Rect> rectangles = getRectangles(bboxes);

    // Perform clustering on the bounding boxes based on color histograms
    cv::Mat featuresMat(rectangles.size(), 32 * 32 * 32, CV_32F);

    for (int i = 0; i < rectangles.size(); i++) {
        cv::Mat roi = image(rectangles[i]);
        cv::Mat feature = calculateColorHistogramFeature(roi);

        // Store the histogram feature in the features matrix
        feature.copyTo(featuresMat.row(i));

        // Normalize the feature
        cv::normalize(featuresMat.row(i), featuresMat.row(i));
    }

    // K-means clustering
    cv::Mat labels, centers;
    cv::kmeans(featuresMat, numClasses, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Paint bounding boxes based on class with transparency
    for (size_t i = 0; i < rectangles.size(); ++i) {
        int label = labels.at<int>(i);
        int teamID = label+1;
        cv::Scalar paintColor = colorMap[teamID];

        // Add colored bounding box to overlay
        cv::rectangle(image, rectangles[i], paintColor, 3);
        cv::rectangle(overlay, rectangles[i], paintColor, -1);

        bboxes[i].setTeamID(teamID);
    }

    // Blend the overlay with the original image
    cv::addWeighted(image, 0.5, overlay, 0.5, 0, overlay);

    return overlay;
}
