//
// Created by Alberto Dorizza
//

#include "../include/ColorFeaturesSegmentator.h"


// ********** PUBLIC METHODS **********
void ColorFeaturesSegmentator::segment(cv::Mat bgrImage, cv::Mat& segmentedImage) const {

    // Convert BGR to YCbCr (Note: channels 0 = Y, channels 1 = Cr, channels 2 = CB)
    cv::Mat yCrCbImage;
    cv::cvtColor(bgrImage, yCrCbImage, cv::COLOR_BGR2YCrCb);

    // Generate Cb-Cr 2D histogram
    cv::Mat CbCrHistogram;
    getNormalizedCbCrHistogram(yCrCbImage, CbCrHistogram);

    // Estimate the 2D pdf through convolution with N-point Gaussian Kernel
    cv::Mat estimated2dPDF;
    estimated2Dpdf(CbCrHistogram, estimated2dPDF);

    // Obtain two independent estimated PDF of Cb and Cr
    std::vector<float> pdfCbArray(estimated2dPDF.rows);
    std::vector<float> pdfCrArray(estimated2dPDF.cols);
    calculateIndependentPDFs(estimated2dPDF, pdfCbArray, pdfCrArray);

    // Find local minima and local maxima of the estimated PDFs
    std::vector<int> indexMinCb, indexMaxCb, indexMinCr, indexMaxCr;
    findLocalMaximaMinima(pdfCbArray, indexMinCb, indexMaxCb);
    findLocalMaximaMinima(pdfCrArray, indexMinCr, indexMaxCr);

    // Generate all start-end points of "blocks" in CbCr plane
    std::vector<cv::Point2i> allPoints;
    generateStartEndBlocksPoint(indexMinCb, indexMinCr, allPoints);

    // Compute (cb, cr, ymean) for all blocks
    std::vector<Block> allBlocks;
    computeAllBlocks(allPoints, indexMaxCb, indexMaxCr, yCrCbImage, allBlocks);

    // Compute final clusters (merge blocks with same color)
    std::vector<Cluster> finalClusters;
    for(cv::Vec3b c : PRINCIPAL_COLORS_RGB) finalClusters.push_back(Cluster(c));
    computeFinalClusters(allBlocks, finalClusters);

    // Compute first segmentation according to the computed clusters
    segmentedImage = cv::Mat(bgrImage.rows, bgrImage.cols, CV_8UC3, cv::Scalar(0)); // change color
    fillSegmentedImage(finalClusters, segmentedImage);

    // merging regions
    mergeSmallRegions(segmentedImage); //(comment for no merging)

    // Display all steps of computation
    /**showImage(bgrImage, "BGR Input Image");
    showImage(yCrCbImage, "YCbCr Input Image");
    showImage(CbCrHistogram, "2D Cb-Cr histogram");
    showImage(estimated2dPDF, "2D estimated PDF");
    displayHistogram(pdfCbArray, "PDF Cb");
    displayHistogram(pdfCrArray, "PDF Cr");
    displayHistogramMinMax(pdfCbArray, indexMinCb, indexMaxCb, "PDF Cb with local minima/maxima");
    displayHistogramMinMax(pdfCrArray, indexMinCr, indexMaxCr, "PDF Cr with local minima/maxima");
    displayBlocksBoundaries(CbCrHistogram, allPoints, indexMinCb, indexMaxCb, indexMinCr, indexMaxCr);
    displayBlocks(CbCrHistogram, allBlocks);
    displayRequantizedImage(bgrImage, allBlocks);
    showImage(segmentedImage, "Final segmentation with region merging");**/
}




// ********** PRIVATE METHODS **********
void ColorFeaturesSegmentator::getNormalizedCbCrHistogram(cv::Mat yCbCrImage, cv::Mat& histogram) const {
    // Define histogram parameters
    int histSize[2] = {256, 256}; // Number of bins for Cb and Cr channel

    float rangeCb[] = {0, 256}; // Range for Cb channel
    float rangeCr[] = {0, 256}; // Range for Cr channel
    const float* hranges[] = { rangeCb, rangeCr };

    int channelsToHistogram[] = {1, 2}; // Cb and Cr channels

    bool uniform = true;
    bool accumulate = false;

    // Calculate 2D histogram for Cb and Cr channels (type: CV_32FC1)
    cv::calcHist(&yCbCrImage, 1, channelsToHistogram, cv::Mat(), histogram,
                 2, histSize, hranges, uniform, accumulate);


    // Normalize the histogram
    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());
}


void ColorFeaturesSegmentator::estimated2Dpdf(cv::Mat hist, cv::Mat& estimated2dPDF) const {
    cv::Mat gaussianKernel(N, N, CV_32F);

    // Generate Gaussian kernel
    for (int k = 0; k < N; ++k) {
        for (int l = 0; l < N; ++l) {
            gaussianKernel.at<float>(k, l) = std::exp(-0.5 * std::pow(alpha / (N / 2.0), 2) * (std::pow(k - ((N - 1) / 2.0), 2) + std::pow(l - ((N - 1) / 2.0), 2)));;
        }
    }

    // Convolve the histogram with the Gaussian kernel
    cv::filter2D(hist, estimated2dPDF, -1, gaussianKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // Normalize the histogram
    cv::normalize(estimated2dPDF, estimated2dPDF, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());
}


void ColorFeaturesSegmentator::calculateIndependentPDFs(cv::Mat estimated2dPDF, std::vector<float>& pdfCbArray, std::vector<float>& pdfCrArray) const {
    // Temp matrix for computing estimated pdf as array after
    cv::Mat pdfCb(1, estimated2dPDF.cols, CV_32F, cv::Scalar(0));
    cv::Mat pdfCr(estimated2dPDF.rows, 1, CV_32F, cv::Scalar(0));

    // Calculate the PDF of Cr by summing along columns
    for (int r = 0; r < estimated2dPDF.rows; ++r) {
        for (int c = 0; c < estimated2dPDF.cols; ++c) {
            pdfCr.at<float>(r, 0) += estimated2dPDF.at<float>(r, c);
        }
    }
    // Normalize the matrix
    cv::normalize(pdfCr, pdfCr, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());


    // Calculate the PDF of Cb by summing along rows
    for (int c = 0; c < estimated2dPDF.cols; ++c) {
        for (int r = 0; r < estimated2dPDF.rows; ++r) {
            pdfCb.at<float>(0, c) += estimated2dPDF.at<float>(r, c);
        }
    }
    // Normalize the matrix
    cv::normalize(pdfCb, pdfCb, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());

    // Fills the resulting arrays and sets values very close to 0 as 0
    for (int r = 0; r < pdfCr.rows; ++r) {
        pdfCrArray.at(r) = pdfCr.at<float>(r, 0);
        if(pdfCrArray.at(r) < 1e-2f) pdfCrArray.at(r) = 0;
    }
    for (int c = 0; c < pdfCb.cols; ++c) {
        pdfCbArray.at(c) = pdfCb.at<float>(0, c);
        if(pdfCbArray.at(c) < 1e-2f) pdfCbArray.at(c) = 0;
    }
}


void ColorFeaturesSegmentator::findLocalMaximaMinima(std::vector<float> arr, std::vector<int>& mn, std::vector<int>& mx) const {
    int n = arr.size();

    // Checking whether the first point is local maxima or minima or none
    if (arr.at(0) > arr.at(1))
        mx.push_back(0);
    else if (arr.at(0) < arr.at(1))
        mn.push_back(0);

    // push into local minima first point different from 0
    for(int i = 1; i < n-1; i++) {
        if(arr.at(i) == 0 && arr.at(i+1) != 0) {
            mn.push_back(i + 1);
            break;
        }
    }

    // Iterating over all points to check local maxima and local minima
    for(int i = 1; i < n - 1; i++) {
        // Condition for local minima
        if ((arr.at(i - 1) > arr.at(i)) && (arr.at(i) < arr.at(i+1)))
            mn.push_back(i);

            // Condition for local maxima
        else if ((arr.at(i-1) < arr.at(i)) && (arr.at(i) > arr.at(i+1)))
            mx.push_back(i);
    }

    // Checking whether the last point is local maxima or minima or none
    if (arr.at(n-1) > arr.at(n - 2))
        mx.push_back(n - 1);
    else if (arr.at(n-1) < arr.at(n - 2))
        mn.push_back(n - 1);

    // push into local minima last point different from 0
    for(int i = n-1; i > 1; i--) {
        if(arr.at(i) == 0 && arr.at(i-1) != 0) {
            mn.push_back(i - 1);
            break;
        }
    }
}


void ColorFeaturesSegmentator::generateStartEndBlocksPoint(std::vector<int> indexMinCb, std::vector<int> indexMinCr, std::vector<cv::Point2i>& allPoints) const {
    for (int i = 0; i < indexMinCb.size()-1; ++i) {
        for (int j = 0; j < indexMinCr.size()-1; ++j) {
            // since the local minima points define the blocks boundaries
            allPoints.push_back(cv::Point2i(indexMinCb.at(i), indexMinCr.at(j)));
            allPoints.push_back(cv::Point2i(indexMinCb.at(i+1), indexMinCr.at(j+1)));
        }
    }
}


void ColorFeaturesSegmentator::computeAllBlocks(std::vector<cv::Point2i> allPoints, std::vector<int> indexMaxCb, std::vector<int> indexMaxCr, cv::Mat yCrCbImage, std::vector<Block>& allBlocks) const {
    int ymean;
    cv::Point2i start, end;
    std::vector<cv::Point2i> pixelCoords;
    cv::Vec3b label;

    for (int mcb: indexMaxCb) {
        for (int mcr: indexMaxCr) {
            for (int i = 0; i < allPoints.size() - 1; ++i) {
                start = allPoints.at(i);
                end = allPoints.at(i+1);
                // check in which rectangle each pair (max_cb, max_cr) is enclosed
                if (mcb > start.x && mcb < end.x && mcr > start.y && mcr < end.y) {
                    // compute all characteristics of the block
                    ymean = computeYmean(yCrCbImage, start, end, pixelCoords);
                    label = computeBlockLabel(start, end);
                    allBlocks.push_back(Block(mcb, mcr, ymean, start, end, pixelCoords, label));

                    // empties the vector
                    pixelCoords.clear();
                }
            }
        }
    }
}


int ColorFeaturesSegmentator::computeYmean(cv::Mat yCrCbImage, cv::Point2i start, cv::Point2i end, std::vector<cv::Point2i>& pixelCoords) const {
    int cumulativeY = 0, n = 0;
    cv::Vec3b p;

    for (int i = 0; i < yCrCbImage.rows; ++i) {
        for (int j = 0; j < yCrCbImage.cols; ++j) {
            p = yCrCbImage.at<cv::Vec3b>(i, j);

            // check if each point in CbCr plane is enclosed in the block
            if(p[2] >= start.x && p[1] >= start.y && p[2] < end.x && p[1] < end.y) { // x = cb, y = cr
                pixelCoords.push_back(cv::Point2i(i, j));
                cumulativeY += yCrCbImage.at<cv::Vec3b>(i, j)[0];
                n++;
            }
        }
    }

    return cumulativeY == 0 ? 0 : cumulativeY  / n;
}


cv::Vec3b ColorFeaturesSegmentator::computeBlockLabel(cv::Point2i start, cv::Point2i end) const {
    int votes[4] = {0, 0, 0, 0}; // red, green, blue, gray
    const int N = sizeof(votes) / sizeof(int);

    for (int cb = start.x; cb < end.x; ++cb) {
        for (int cr = start.y; cr < end.y; ++cr) {
            if (0.392 * (cb - 128) + 2.409 * (cr - 128) > 0 &&
                2.017 * (cb - 128) - 1.596 * (cr - 128) < 0) // r>g && r>b => red
                votes[0]++;

            if (0.392 * (cb - 128) + 2.409 * (cr - 128) < 0 &&
                -2.409 * (cb - 128) - 0.813 * (cr - 128) > 0) // g>r && g>b => green
                votes[1]++;

            if (2.017 * (cb - 128) - 1.596 * (cr - 128) > 0 &&
                -2.409 * (cb - 128) - 0.813 * (cr - 128) < 0) // b>r && b>g => blue
                votes[2]++;

            if (std::sqrt(std::pow(cb - 128, 2) + std::pow(cr - 128, 2)) < TH) // => gray
                votes[3]++;
        }
    }

    int colorIdx = std::distance(votes, std::max_element(votes, votes + N)); // index of max element
    return PRINCIPAL_COLORS_RGB.at(colorIdx);
}


void ColorFeaturesSegmentator::computeFinalClusters(std::vector<Block> allBlocks, std::vector<Cluster>& finalClusters) const {
    for (Block b: allBlocks) {
        for (Cluster &c: finalClusters) {
            if (cv::norm(b.getLabel(), c.getLabel(), cv::DIST_L2) == 0) { // check if block label and cluster label are the same
                for (cv::Point2i pixel: b.getPixelsCoords())
                    c.addPixel(pixel); // add all points of the blocks in the cluster with the same label
            }
        }
    }
}


void ColorFeaturesSegmentator::fillSegmentedImage(std::vector<Cluster> finalClusters, cv::Mat& segmentedImage) const {
    for(Cluster c : finalClusters) {
        for(cv::Point2i p : c.getPixelsCoords()) {
            segmentedImage.at<cv::Vec3b>(p.x, p.y) = c.getLabel();
        }
    }
}

void ColorFeaturesSegmentator::mergeSmallRegions(cv::Mat& segmentedImage) const {
    for(cv::Vec3b target : PRINCIPAL_COLORS_RGB) {
        // Create a binary mask for the target color
        cv::Mat colorMask;
        cv::inRange(segmentedImage, target, target, colorMask);

        // Create a label matrix to store connected component labels
        cv::Mat labels, stats, centroids;
        int numComponents = cv::connectedComponentsWithStats(colorMask, labels, stats, centroids);

        // Display the number of connected components (including background)
        // Iterate through each connected component and draw bounding boxes
        for (int label = 1; label < numComponents; label++) {
            // Create a mask for the current component
            cv::Mat componentMask = (labels == label);

            // Extract the bounding box of the component
            cv::Rect boundingBox(stats.at<int>(label, cv::CC_STAT_LEFT),
                                 stats.at<int>(label, cv::CC_STAT_TOP),
                                 stats.at<int>(label, cv::CC_STAT_WIDTH),
                                 stats.at<int>(label, cv::CC_STAT_HEIGHT));


            cv::Mat rectImage (colorMask.size(), CV_8UC1);
            cv::rectangle(rectImage, boundingBox, cv::Scalar(255), -1);

            // Create a mask for the ROI (white pixels inside the ROI)
            cv::Mat roiMask(colorMask.size(), CV_8UC1, cv::Scalar(0));
            colorMask.copyTo(roiMask, rectImage);

            int minRegionSize = (segmentedImage.rows * segmentedImage.cols) * 0.0005;
            //minRegionSize = 100; // other possible fixed parameters choices
            //std::cout << "minRegionSize" << minRegionSize << std::endl;
            if (cv::countNonZero(roiMask) < minRegionSize) {
                // Calculate the center of the rectangle
                cv::Point center(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
                // Calculate the axes of the ellipse (half of the width and height of the rectangle)
                int majorAxis = (boundingBox.width / 2) + 25;
                int minorAxis = (boundingBox.height / 2) + 25;

                // Create a new image to draw the ellipse
                cv::Mat imageWithEllipse = cv::Mat::zeros(colorMask.size(), CV_8UC1); // Create a black image

                // Draw the enclosing ellipse on the new image
                cv::ellipse(imageWithEllipse, center, cv::Size(majorAxis, minorAxis), 0, 0, 360, cv::Scalar(255, 255, 255), -1);

                cv::Mat diff = cv::Mat::zeros(segmentedImage.size(), CV_8UC1);
                diff = imageWithEllipse - roiMask;

                //Define your destination image
                cv::Mat roi = cv::Mat::zeros(segmentedImage.size(), segmentedImage.type());
                segmentedImage.copyTo(roi, diff);

                // Initialize counters for each specific color
                std::vector<int> colorCounts(4, 0);

                int i = 0;
                for(cv::Vec3b color : PRINCIPAL_COLORS_RGB) {
                    cv::Mat mask;
                    inRange(roi, color, color, mask);
                    colorCounts.at(i++) = cv::countNonZero(mask);
                }

                // Find the color with the maximum count (dominant color)
                int maxCountIndex = std::distance(colorCounts.begin(),
                                                  std::max_element(colorCounts.begin(), colorCounts.end()));

                // Print the dominant color
                cv::Vec3b dominantColor = PRINCIPAL_COLORS_RGB[maxCountIndex];

                segmentedImage.setTo(dominantColor, roiMask);
            }
        }
    }
}




// ********** UTILS METHODS **********
void ColorFeaturesSegmentator::showImage(cv::Mat img, std::string name) const {
    // Create a window and display the input image
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, img);
    cv::waitKey(0);
}

void ColorFeaturesSegmentator::displayHistogram(std::vector<float> vec, std::string name) const {
    // Create a window for the histogram
    cv::namedWindow("Histogram for "+name, cv::WINDOW_NORMAL);

    // Create Mat objects for the histogram images
    cv::Mat histImage(vec.size(), vec.size(), CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw the histogram
    for (int i = 0; i < vec.size(); ++i) {
        int value = cvRound(vec[i] * 255); // Scale to 0-255 range

        // Draw a line at each index with height "value"
        cv::line(histImage, cv::Point(i, histImage.rows), cv::Point(i, histImage.rows - value), cv::Scalar(0, 0, 0), 1);
    }

    // Display the histograms in their respective windows
    cv::imshow("Histogram for " + name, histImage);
    cv::waitKey(0);
}

void ColorFeaturesSegmentator::displayHistogramMinMax(std::vector<float> vec, std::vector<int> indexMin, std::vector<int> indexMax, std::string name) const {
    // Create a window for the histogram
    cv::namedWindow("Histogram for " + name, cv::WINDOW_NORMAL);

    // Create Mat objects for the histogram images
    cv::Mat histImage(vec.size(), vec.size(), CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw the histogram
    for (int i = 0; i < vec.size(); ++i) {
        int value = cvRound(vec[i] * 255); // Scale to 0-255 range

        if (std::find(indexMax.begin(), indexMax.end(), i) != indexMax.end()) {
            cv::line(histImage, cv::Point(i, histImage.rows), cv::Point(i, histImage.rows - value),
                     cv::Scalar(255, 0, 0), 2);
        } else if (std::find(indexMin.begin(), indexMin.end(), i) != indexMin.end()) {
            cv::line(histImage, cv::Point(i, histImage.rows), cv::Point(i, histImage.rows - value),
                     cv::Scalar(0, 0, 255), 2);
        } else {
            cv::line(histImage, cv::Point(i, histImage.rows), cv::Point(i, histImage.rows - value),
                     cv::Scalar(0, 0, 0), 2);
        }

        // Draw a line at each index with height "value"
        cv::line(histImage, cv::Point(i, histImage.rows), cv::Point(i, histImage.rows - value), cv::Scalar(0, 0, 0), 1);
    }

    // Display the histograms in their respective windows
    cv::imshow("Histogram for " + name, histImage);
    cv::waitKey(0);
}


void ColorFeaturesSegmentator::displayBlocksBoundaries(cv::Mat CbCrHistogram, std::vector<cv::Point2i> allPoints, std::vector<int> indexMinCb, std::vector<int> indexMaxCb, std::vector<int> indexMinCr, std::vector<int> indexMaxCr) const {
    cv::Mat blocksBoundaries (CbCrHistogram.rows, CbCrHistogram.cols, CV_8UC3, cv::Vec3b(255, 255, 255));
    for (int maxIndexCb: indexMaxCb) {
        for (int maxIndexCr: indexMaxCr) {
            blocksBoundaries.at<cv::Vec3b>(maxIndexCb, maxIndexCr) = cv::Vec3b(0, 0, 255); // maxima points in red
        }
    }

    for (int minIndexCb: indexMinCb) {
        for (int i = indexMinCr.at(0); i < indexMinCr.at(indexMinCr.size() - 1); ++i) {
            blocksBoundaries.at<cv::Vec3b>(minIndexCb, i) = cv::Vec3b(0, 0, 0);
        }
    }

    for (int minIndexCr: indexMinCr) {
        for (int i = indexMinCb.at(0); i < indexMinCb.at(indexMinCb.size() - 1); ++i) {
            blocksBoundaries.at<cv::Vec3b>(i, minIndexCr) = cv::Vec3b(0, 0, 0);
        }
    }

    for (cv::Point2i p: allPoints) {
        blocksBoundaries.at<cv::Vec3b>(p.x, p.y) = cv::Vec3b(0, 255, 0); // start-end point in green
    }
    showImage(blocksBoundaries, "blocksBoundaries");
}


void ColorFeaturesSegmentator::displayBlocks(cv::Mat CbCrHistogram, std::vector<Block> allBlocks) const {
    cv::Mat blocksRes = cv::Mat(CbCrHistogram.rows, CbCrHistogram.cols, CV_8UC3, cv::Vec3b(255, 255, 255));

    for (Block b: allBlocks) {
        for (int i = b.getStart().x; i < b.getEnd().x; ++i) {
            for (int j = b.getStart().y; j < b.getEnd().y; ++j) {
                // set all pixels in the blocks with the Y Cb Cr of the maximum of that block
                blocksRes.at<cv::Vec3b>(i, j)[0] = b.getYmean();
                blocksRes.at<cv::Vec3b>(i, j)[1] = b.getCrMax();
                blocksRes.at<cv::Vec3b>(i, j)[2] = b.getCbMax();
            }
        }
        blocksRes.at<cv::Vec3b>(b.getCbMax(), b.getCrMax()) = cv::Vec3b(0, 0, 0); // maximum in black
    }
    showImage(blocksRes, "Resulted blocks");

    cv::Mat blocksResBgr = blocksRes.clone();
    cv::cvtColor(blocksRes, blocksResBgr, cv::COLOR_YCrCb2BGR);
    showImage(blocksResBgr, "Resulted blocks BGR");
}


void ColorFeaturesSegmentator::displayRequantizedImage(cv::Mat bgrImage, std::vector<Block> allBlocks) const {
    cv::Mat requantizedImage = cv::Mat(bgrImage.rows, bgrImage.cols, CV_8UC3, cv::Vec3b(255, 255, 255));
    for (Block b: allBlocks) {
        for (cv::Point2i p: b.getPixelsCoords()) {
            requantizedImage.at<cv::Vec3b>(p.x, p.y)[0] = b.getYmean();
            requantizedImage.at<cv::Vec3b>(p.x, p.y)[1] = b.getCrMax();
            requantizedImage.at<cv::Vec3b>(p.x, p.y)[2] = b.getCbMax();
        }
    }
    cv::cvtColor(requantizedImage, requantizedImage, cv::COLOR_YCrCb2BGR);
    showImage(requantizedImage, "Re-quantized image BGR");
}


std::string ColorFeaturesSegmentator::type2str(int type) const {
    // returned string
    std::string r;

    // number of channels and depth of each
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    // mapping
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    // for standard notation
    r += "C";
    r += (chans+'0');

    return r;
}
