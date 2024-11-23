//
// Created by Dario Mameli
//

#include "../include/detection_segmentation.h"


std::vector<Player> extractPlayersFromSegments(const std::string& masks_directory_path, const std::filesystem::path& image_path) {

    /// INITIALIZATION

    std::string imageFilename = image_path.filename().string();
    std::string imageName = imageFilename.substr(0, imageFilename.length()-4);

    std::vector<Player> players {};
    cv::Mat image = cv::imread(image_path.string());


    /// SEGMENTS EXTRACTION

    try {
        // Search for all the masks
        for (const auto& entry : std::filesystem::directory_iterator(masks_directory_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {

                std::string maskFilename = entry.path().filename().string();
                std::string maskName = maskFilename.substr(0, maskFilename.length()-4);

                // Consider only the masks which refer to the given image
                if(maskName.substr(0, maskName.find('_'))==imageName) {
                    cv::Mat mask = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                    // For player segmentation build the Player vector
                    cv::Mat binaryMask = mask != 0;
                    cv::Mat roi;
                    image.copyTo(roi, binaryMask);
                    /*
                    cv::imshow("binaryMask", binaryMask);
                    cv::imshow("roi", roi);
                    cv::waitKey(0);
                     */
                    players.emplace_back(roi);

                    // For playing field segmentation update the mask with pixels without segments
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return players;
}

void classifyPlayers(std::vector<Player>& players, const std::filesystem::path& image_path, const cv::Mat& imagePlayingField_segmented, const std::string& masksPath, const std::string& bboxImagesPath) {

    /// SEGMENTS CLASSIFICATION

    // Mapping cluster label to class color
    std::map<int, cv::Scalar> colorMap = {
            {0, cv::Scalar(255, 0, 0)}, // Blue -> Team 1
            {1, cv::Scalar(0, 0, 255)} // Red -> Team 2
    };

    // Define gray levels for the classes. Also valid mapping from label cluster to teamID
    std::map<int, uchar> grayMap = {
            {0, 1}, // 1 -> Team 1
            {1, 2} // 2 -> Team 2
    };

    // Perform clustering on the bounding boxes based on color histograms
    cv::Mat featuresMat(players.size(), 32 * 32 * 32, CV_32F);

    for (int i = 0; i < players.size(); i++) {
        cv::Mat feature = calculateColorHistogramFeature(players[i].getSegment());

        // Store the histogram feature in the features matrix
        feature.copyTo(featuresMat.row(i));

        // Normalize the feature
        cv::normalize(featuresMat.row(i), featuresMat.row(i));
    }

    // K-means clustering
    cv::Mat labels, centers;
    cv::kmeans(featuresMat, 2, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2),
               3, cv::KMEANS_PP_CENTERS, centers);


    /// INITIALIZATION FOR SAVING PURPOSES
    // Original image
    cv::Mat image = cv::imread(image_path.string());
    // Image where to paint the colored segments
    cv::Mat imageSegmentation = imagePlayingField_segmented.clone();
    // Image where to paint the grayscale segments
    cv::Mat imageSegmentationGray;
    cv::cvtColor(imageSegmentation, imageSegmentationGray, cv::COLOR_BGR2GRAY);

    // Paint the background and playing field of the gray segmented image with the correct values.
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // If it's a playing field pixel (green)
            if (imageSegmentation.at<cv::Vec3b>(y, x)==cv::Vec3b(0,255,0))
                imageSegmentationGray.at<uchar>(y, x) = 3;
            // Otherwise it's background
            else
                imageSegmentationGray.at<uchar>(y, x) = 0;
        }
    }

    std::vector<BoundingBox> bboxes;


    /// MASKS AND TXT CREATION

    // Paint bounding boxes and segments of each player based on class color
    for (size_t i = 0; i < players.size(); ++i) {

        // If player is not in the playing field then it's not a player and shouldn't be considered
        if(!players[i].isInPlayingField(imageSegmentation))
            continue;

        int label = labels.at<int>(i);
        cv::Scalar paintColor = colorMap[label];

        /// DETECTION

        // Extract the rectangle from the bounding box of the player
        cv::Rect r = players[i].getBoundingBox().getRectangle();

        // Set the teamID accordingly
        players[i].setTeamID(grayMap[label]);

        // Add bounding box to the vector for saving
        bboxes.push_back(players[i].getBoundingBox());

        /// SEGMENTATION

        // Create a mask where the segment is not zero (i.e., it exists)
        cv::Mat segmentGray;
        cv::cvtColor(players[i].getSegment(), segmentGray, cv::COLOR_BGR2GRAY);
        cv::Mat segmentMask = segmentGray > 0;

        // cv::imshow("segmentMask", segmentMask);
        // cv::waitKey(0);

        // Create color to assign to segmented pixels
        cv::Vec3b color;
        color[0] = paintColor[0];
        color[1] = paintColor[1];
        color[2] = paintColor[2];

        // Paint on top of the colored segmentation image and the gray one with the players
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                if (segmentMask.at<uchar>(y, x)) {
                    imageSegmentation.at<cv::Vec3b>(y, x) = color;
                    imageSegmentationGray.at<uchar>(y, x) = grayMap[label];
                }
            }
        }
    }

    /// SAVING

    std::string image_filename = image_path.filename().string();
    std::string image_name = image_filename.substr(0, image_filename.length()-4);

    // Check if the directory already exists
    if (!std::filesystem::exists(masksPath)) {
        // Create the directory if it doesn't exist
        std::filesystem::create_directories(masksPath);
    }

    // Check if the directory already exists
    if (!std::filesystem::exists(bboxImagesPath)) {
        // Create the directory if it doesn't exist
        std::filesystem::create_directories(bboxImagesPath);
    }

    // Bounding Boxes
    std::string output_BBoxes_path = masksPath+image_name+"_bb.txt";
    saveBoundingBoxes(output_BBoxes_path, bboxes);
    saveImageDetections(image_path, bboxImagesPath, bboxes);

    // Segmentation Masks
    std::string output_colorMask_path = masksPath+image_name+"_color.png";
    std::string output_grayMask_path = masksPath+image_name+"_bin.png";
    cv::imwrite(output_colorMask_path, imageSegmentation);
    std::cout << "Saved " << output_colorMask_path << std::endl;
    cv::imwrite(output_grayMask_path, imageSegmentationGray);
    std::cout << "Saved " << output_grayMask_path << std::endl;


    /// DISPLAYING

    /*
    cv::imshow("Segmentation", imageSegmentation);
    cv::imshow("Segmentation gray", imageSegmentationGray);
    cv::waitKey(0);
     */

}

void detectionANDsegmentation(const std::string& masks_directory_path, const std::filesystem::path& image_path, const cv::Mat& segmented_playingField_image, const std::string& masksPath, const std::string& bboxImagesPath) {

    /// PLAYING FIELD

    // Compress the segmented regions into only playing field and background

    cv::Mat imagePlayingFieldSegmented = kmeansSegmentationPlayingField(segmented_playingField_image);



    /// PLAYERS

    auto beg = std::chrono::high_resolution_clock::now(); // Time check
    std::vector<Player> players = extractPlayersFromSegments(masks_directory_path, image_path); // Get all the segments and build the corresponding players

    // If there are players then proceed as normal
    if(!players.empty()) {
        classifyPlayers(players, image_path, imagePlayingFieldSegmented, masksPath,
                        bboxImagesPath); // Classify each player
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
        std::cout << "Detection and segmentation of players inference time for " << image_path.filename() << ": "
                  << duration.count() << " ms" << std::endl;
    } else {
        /// NO PLAYER SEGMENTS FOUND

        // Initialization

        cv::Mat imageSegmentation = imagePlayingFieldSegmented.clone();
        // Image where to paint the grayscale segments
        cv::Mat imageSegmentationGray;
        cv::cvtColor(imageSegmentation, imageSegmentationGray, cv::COLOR_BGR2GRAY);

        // Paint the background and playing field of the gray segmented image with the correct values.
        for (int y = 0; y < imageSegmentation.rows; ++y) {
            for (int x = 0; x < imageSegmentation.cols; ++x) {
                // If it's a playing field pixel (green)
                if (imageSegmentation.at<cv::Vec3b>(y, x)==cv::Vec3b(0,255,0))
                    imageSegmentationGray.at<uchar>(y, x) = 3;
                    // Otherwise it's background
                else
                    imageSegmentationGray.at<uchar>(y, x) = 0;
            }
        }

        // Save segmented playing field and background only
        std::string image_filename = image_path.filename().string();
        std::string image_name = image_filename.substr(0, image_filename.length()-4);
        std::string output_colorMask_path = masksPath+image_name+"_color.png";
        std::string output_grayMask_path = masksPath+image_name+"_bin.png";
        cv::imwrite(output_colorMask_path, imageSegmentation);
        std::cout << "Saved " << output_colorMask_path << std::endl;
        cv::imwrite(output_grayMask_path, imageSegmentationGray);
        std::cout << "Saved " << output_grayMask_path << std::endl;

        throw SegmentsNotFoundException("No segments were found for " + image_path.filename().string());
    }
}