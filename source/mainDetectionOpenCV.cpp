//
// Created by Sara Farris
//

#include "../include/performance_measurement.h"
#include "../include/ColorFeaturesSegmentator.h"
#include "../include/detection.h"
#include <chrono>

int main(int argc,char**argv) {

    std::map<std::string, std::string> argMap;
    int choice = -1; // Initialize choice with an invalid value

    // Iterate through the command-line arguments and save them to the map
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        // Check if the argument starts with "--"
        if (arg.find("--") == 0) {
            // Remove "--" from the argument
            arg = arg.substr(2);
            // Split the argument into name and value using '=' as a delimiter
            size_t pos = arg.find('=');
            if (pos != std::string::npos) {
                std::string argName = arg.substr(0, pos);
                std::string argValue = arg.substr(pos + 1);
                argMap[argName] = argValue;
            } else {
                std::cerr << "Invalid argument format: " << argv[i] << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Invalid argument format: " << argv[i] << std::endl;
            return 1;
        }
    }

    // Check if the expected arguments are provided
    if (argMap.size() != 3 ||
        argMap.find("dataset_path") == argMap.end() ||
        argMap.find("groundTruthMasks") == argMap.end() ||
        argMap.find("choice") == argMap.end()) {
        std::cerr << "Usage: " << argv[0] << " --dataset_path=<value> --groundTruthMasks=<value> --choice=<0-2>" << std::endl;
        return 1;
    }

    // Retrieve and validate the choice value
    try {
        choice = std::stoi(argMap["choice"]);
        if (choice < 0 || choice > 2) {
            std::cerr << "Invalid choice value. It must be between 0 and 2." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid choice format: " << argMap["choice"] << std::endl;
        return 1;
    }

    /// INPUT PATHS
    const std::string dataset_path = argMap["dataset_path"];
    const std::string groundTruthMasks = argMap["groundTruthMasks"];

    ///************** ALGORITHM CHOICE****************
    /**
     const int choice = 2; // change to test the different algorithms.
    /* As in "detection.h"->void detection(...) doc:
    * choice=0) hogDescriptorSVM
    * choice=1) haarFeaturesCascadeDetector
    * choice=2) contoursHogSVM
    */

    std::map<int, std::string> algorithm {
            {0,"HOG + SVM"},
            {1,"Haar Cascade"},
            {2,"Contours + HOG + SVM"}
    };


    //*********** PATHS ***********************
    /**
    /// INPUT PATHS
    const std::string dataset_path = "../Sport_scene_dataset/Images/";
    const std::string groundTruthMasks = "../Sport_scene_dataset/Masks/";
    **/


    /// OUTPUT PATHS
    const std::string masksPath = "../Masks/";
    const std::string bboxImagesPath = "../ImagesDetection/";
    const std::string groundTruthDetectionsPath = "../Sport_scene_dataset/groundTruthDetectionImages/";
    const std::string classifiedImagesPath = "../runs/"+algorithm[choice]+"/Images/";
    const std::string classifiedLabelsPath = "../runs/"+algorithm[choice]+"/BBoxes/";
    const std::string performanceMeasurementsDirectory = "../performance_measurements/";
    const std::string mapPath = performanceMeasurementsDirectory+"map.txt";


    // Build directories if they don't exist
    try {
        if (!std::filesystem::exists(masksPath))
            std::filesystem::create_directory(masksPath);
        if (!std::filesystem::exists(bboxImagesPath))
            std::filesystem::create_directory(bboxImagesPath);

        if (!std::filesystem::exists(groundTruthDetectionsPath))
            std::filesystem::create_directories(groundTruthDetectionsPath);
        if (!std::filesystem::exists(classifiedImagesPath))
            std::filesystem::create_directories(classifiedImagesPath);
        if (!std::filesystem::exists(classifiedLabelsPath))
            std::filesystem::create_directories(classifiedLabelsPath);

        if (!std::filesystem::exists(classifiedLabelsPath))
            std::filesystem::create_directory(classifiedLabelsPath);
        if (!std::filesystem::exists(performanceMeasurementsDirectory))
            std::filesystem::create_directory(performanceMeasurementsDirectory);
    } catch (const std::exception& e) {
        std::cerr << "Error creating folders: " << e.what() << std::endl;
    }


    //*********** OUTPUT FILES FOR PERFORMANCES *********
    /// mAP
    std::ofstream outputMapFile(mapPath);
    // Write the output mAP file
    if (!outputMapFile.is_open()) {
        // Close the output files
        outputMapFile.close();
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }


    //*********** OPENCV DETECTION *********
    std::cout << "********************************************* OPENCV DETECTION *********************************************" << std::endl;

    outputMapFile << algorithm[choice] << std::endl;

    std::vector<float> mapVector1;

    auto begTot = std::chrono::high_resolution_clock::now();
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                auto begIter = std::chrono::high_resolution_clock::now();

                /// DETECTION

                std::string imagePath = entry.path().string();
                std::string imageFilename = entry.path().filename().string();
                std::string imageName = imageFilename.substr(0, imageFilename.length()-4);

                std::cout << "\nImage: " << imageFilename << std::endl;

                // Save ground truth detections as images for comparison
                saveImageDetections(entry.path(), groundTruthDetectionsPath, readBoundingBoxes(groundTruthMasks+imageName+"_bb.txt"));

                std::string txtFilename = imageFilename.substr(0,imageFilename.length()-4)+"_bb.txt";
                std::string txtPath = classifiedLabelsPath+txtFilename;

                cv::Mat imgOriginal = cv::imread(imagePath);

                std::vector<cv::Mat> augmentedImages = preprocessing(imgOriginal);

                cv::Mat imgDetection = imgOriginal.clone();

                std::vector<BoundingBox> bboxes;

                detection(&augmentedImages, imgDetection, bboxes, choice);

                // If bounding boxes were not found
                if(bboxes.empty()) {
                    std::cout << "No bounding boxes were detected.." << std::endl;
                    float map = 0;
                    mapVector1.push_back(map);
                    outputMapFile << imageFilename << "\t" << map << std::endl;
                    auto endIter = std::chrono::high_resolution_clock::now();
                    auto iterDuration = std::chrono::duration_cast<std::chrono::seconds>(endIter - begIter);
                    std::cout << "Detection inference time for " << imageFilename << ": " << iterDuration.count() << " s" << std::endl;
                    continue;
                }

                saveBoundingBoxes(txtPath, bboxes);

                saveImageDetections(entry.path(), classifiedImagesPath, bboxes);

                /// CLASSIFICATION ADJUSTMENT

                if(adjustTeamIDAccordingToMAP(imagePath, txtPath, groundTruthMasks+txtFilename)) {
                    std::cout << "Switched teamIDs..." << std::endl;
                    saveImageDetections(entry.path(), classifiedImagesPath, readBoundingBoxes(classifiedLabelsPath+imageName+"_bb.txt"));
                }

                /// PERFORMANCE MEASUREMENT

                float map = mAP(imagePath, txtPath, groundTruthMasks+txtFilename);

                mapVector1.push_back(map);

                outputMapFile << imageFilename << "\t" << map << std::endl;

                auto endIter = std::chrono::high_resolution_clock::now();
                auto iterDuration = std::chrono::duration_cast<std::chrono::seconds>(endIter - begIter);
                std::cout << "Detection inference time for " << imageFilename << ": " << iterDuration.count() << " s" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        // Close the output files
        outputMapFile.close();
        std::cerr << "Error: " << e.what() << std::endl;
    }
    auto endTot = std::chrono::high_resolution_clock::now();
    auto totDuration = std::chrono::duration_cast<std::chrono::minutes>(endTot - begTot);
    std::cout << "\n" << algorithm[choice] << " total inference time: " << totDuration.count() << " min" << std::endl << std::endl << std::endl;

    double sum1 = 0;
    // Calculate the sum of map values
    for (float number : mapVector1) {
        sum1 += number;
    }
    // Calculate the mean
    double mean1 = sum1 /mapVector1.size();
    outputMapFile << "Mean mAP " << mean1 << std::endl << std::endl;

    // Close the output files
    outputMapFile.close();
}