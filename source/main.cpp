//
// Created by Sara Farris
//

#include "../include/detection_segmentation.h"
#include "../include/performance_measurement.h"
#include "../include/ColorFeaturesSegmentator.h"
#include <chrono>


int main(int argc,char**argv) {
    std::map<std::string, std::string> argMap;

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

    // Check if all expected arguments are provided
    if (argMap.size() != 3 ||
        argMap.find("dataset_path") == argMap.end() ||
        argMap.find("groundTruthMasks") == argMap.end() ||
        argMap.find("segmentedMasksPath_pretrained") == argMap.end()) {
        std::cerr << "Usage: " << argv[0] << " --dataset_path=<value> --groundTruthMasks=<value> --segmentedMasksPath_pretrained=<value>" << std::endl;
        return 1;
    }

    /**
    //*********** PATHS ***********************
    /// INPUT PATHS
    const std::string dataset_path = "../Sport_scene_dataset/Images/";
    const std::string groundTruthMasks = "../Sport_scene_dataset/Masks/";
    const std::string segmentedMasksPath_pretrained = "../runs/segment/predict/masks/";
    **/

    /// INPUT PATHS
    const std::string dataset_path = argMap["dataset_path"];
    const std::string groundTruthMasks = argMap["groundTruthMasks"];
    const std::string segmentedMasksPath_pretrained = argMap["segmentedMasksPath_pretrained"];

    /// OUTPUT PATHS
    const std::string masksPath = "../Masks/";
    const std::string bboxImagesPath = "../ImagesDetection/";
    const std::string groundTruthDetectionsPath = "../Sport_scene_dataset/groundTruthDetectionImages/";
    const std::string performanceMeasurementsDirectory = "../performance_measurements/";
    const std::string mapPath = performanceMeasurementsDirectory+"map.txt";
    const std::string miouPath = performanceMeasurementsDirectory+"miou.txt";

    // Display the saved values for verification
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "groundTruthMasks: " << groundTruthMasks << std::endl;
    std::cout << "segmentedMasksPath_pretrained: " << segmentedMasksPath_pretrained << std::endl;

    // Build directories if they don't exist
    try {
        if (std::filesystem::exists(masksPath)) {
            for (const auto& entry : std::filesystem::directory_iterator(masksPath)) {
                std::filesystem::remove_all(entry.path()); // Remove all files and subdirectories
            }
        } else {
            std::filesystem::create_directory(masksPath);
        }

        if (std::filesystem::exists(bboxImagesPath)) {
            for (const auto& entry : std::filesystem::directory_iterator(bboxImagesPath)) {
                std::filesystem::remove_all(entry.path()); // Remove all files and subdirectories
            }
        } else {
            std::filesystem::create_directory(bboxImagesPath);
        }

        if (std::filesystem::exists(groundTruthDetectionsPath)) {
            for (const auto& entry : std::filesystem::directory_iterator(groundTruthDetectionsPath)) {
                std::filesystem::remove_all(entry.path()); // Remove all files and subdirectories
            }
        } else {
            std::filesystem::create_directory(groundTruthDetectionsPath);
        }

        if (std::filesystem::exists(performanceMeasurementsDirectory)) {
            for (const auto& entry : std::filesystem::directory_iterator(performanceMeasurementsDirectory)) {
                std::filesystem::remove_all(entry.path()); // Remove all files and subdirectories
            }
        } else {
            std::filesystem::create_directory(performanceMeasurementsDirectory);
        }

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
    /// mIoU
    std::ofstream outputMiouFile(miouPath);
    // Write the output mIOU file
    if (!outputMiouFile.is_open()) {
        // Close the output files
        outputMapFile.close();
        outputMiouFile.close();
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }


    //*********** YOLO METHOD ****************
    std::cout << "********************************************* MAIN *********************************************" << std::endl;

    outputMapFile << "MAIN" << std::endl;
    outputMiouFile << "MAIN" << std::endl;

    std::vector<float> mapVector2;
    std::vector<float> miouVector;

    ColorFeaturesSegmentator cfs(9, 2.5, 11);

    auto begTot1 = std::chrono::high_resolution_clock::now();
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                auto begIter = std::chrono::high_resolution_clock::now();

                std::string imageFileName = entry.path().filename().string();
                std::string imageName = imageFileName.substr(0, imageFileName.length()-4);

                std::cout << "\nImage: " << imageFileName << std::endl;

                // Save ground truth detections as images for comparison
                saveImageDetections(entry.path(), groundTruthDetectionsPath, readBoundingBoxes(groundTruthMasks+imageName+"_bb.txt"));

                /// PLAYING FIELD SEGMENTATION

                cv::Mat imgOriginal = cv::imread(entry.path().string());
                cv::Mat segmentedImage (imgOriginal.size(), imgOriginal.type(), cv::Scalar(0,0,0));
                cfs.segment(imgOriginal, segmentedImage);

                /// DETECTION + SEGMENTATION

                bool segmentsFound = true;
                try {
                    detectionANDsegmentation(segmentedMasksPath_pretrained, entry.path(), segmentedImage, masksPath,
                                             bboxImagesPath);

                    /// CLASSIFICATION ADJUSTMENT

                    if(adjustTeamIDAccordingToMAP(entry.path().string(), masksPath+imageName+"_bb.txt", groundTruthMasks+imageName+"_bb.txt")) {
                        std::cout << "Switched teamIDs..." << std::endl;
                        saveImageDetections(entry.path(), bboxImagesPath, readBoundingBoxes(masksPath+imageName+"_bb.txt"));
                    }
                } catch(const SegmentsNotFoundException& e) {

                    std::cout << e.what() << std::endl;

                    segmentsFound = false;
                }

                /// PERFORMANCE MEASUREMENT

                float map = 0; // default segments not found
                if(segmentsFound) // if segments were found
                    map = mAP(entry.path().string(), masksPath+imageName+"_bb.txt", groundTruthMasks+imageName+"_bb.txt");
                //std::cout << "mAP for " << imageFileName << " : " << map << std::endl;
                float miou = mIoU(masksPath+imageName+"_bin.png", groundTruthMasks+imageName+"_bin.png");
                //std::cout << "mIOU for " << imageFileName << " : " << miou << std::endl;

                mapVector2.push_back(map);
                miouVector.push_back(miou);

                outputMapFile << imageFileName << "\t" << map << std::endl;
                outputMiouFile << imageFileName << "\t" << miou << std::endl;

                auto endIter = std::chrono::high_resolution_clock::now();
                auto iterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endIter - begIter);
                std::cout << "Total detection and segmentation inference time for " << entry.path().filename() << ": " << iterDuration.count() << " ms" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        // Close the output files
        outputMapFile.close();
        outputMiouFile.close();
        std::cerr << "Error: " << e.what() << std::endl;
    }
    auto endTot1 = std::chrono::high_resolution_clock::now();
    auto totDurationSec = std::chrono::duration_cast<std::chrono::seconds>(endTot1 - begTot1);
    std::cout << "\nMAIN METHOD total inference time: " << totDurationSec.count() << " s" << std::endl;

    double sum2 = 0;
    // Calculate the sum of mAP values
    for (float number : mapVector2) {
        sum2 += number;
    }
    // Calculate the mean
    double mean2 = sum2 /mapVector2.size();
    outputMapFile << "MeanMAP " << mean2 << std::endl;

    sum2 = 0;
    // Calculate the sum of mIoU values
    for (float number : miouVector) {
        sum2 += number;
    }
    // Calculate the mean
    mean2 = sum2 /miouVector.size();
    outputMiouFile << "MeanMIOU " << mean2 << std::endl;

    // Close the output files
    outputMapFile.close();
    outputMiouFile.close();
}