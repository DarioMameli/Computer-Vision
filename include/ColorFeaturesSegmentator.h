//
// Created by Alberto Dorizza
//

#ifndef SPORTVIDEOANALYSIS_SADVISION_COLORFEATURESSEGMENTATOR_H
#define SPORTVIDEOANALYSIS_SADVISION_COLORFEATURESSEGMENTATOR_H

#include <string>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

class ColorFeaturesSegmentator {

    // Utility vector of main segmentation colours
    const std::vector<cv::Vec3b> PRINCIPAL_COLORS_RGB = {cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255), cv::Vec3b(189, 189, 189)};

public:

    /**
     * Constructor of ColorFeaturesSegmentator: setting the parameters of N-point Gaussian window and the gray color threshold value
     * @param N_gaussian kernel parameter
     * @param alpha_gaussian kernel parameter
     * @param TH_gray the gray color threshold value
     */
    ColorFeaturesSegmentator(float N_gaussian, float alpha_gaussian, float TH_gray) :
            N{N_gaussian}, alpha{alpha_gaussian}, TH{TH_gray} {}


    // ********** PUBLIC METHODS **********
    /**
     * Main method that receives a BGR image and returns its segmentation into the 4 main colours defined in the constant vector PRINCIPAL_COLORS_RGB.
     * @param bgrImage input image
     * @param segmentedImage resulting image segmented with 4 main colours
     */
    void segment(cv::Mat bgrImage, cv::Mat& segmentedImage) const;

    // Getters
    float getN() const { return N; }
    float getAlpha() const { return alpha; }
    float getTh() const { return TH; }

    // Setters (for changing segmentator parameters)
    void setN(float n) { N = n; }
    void setAlpha(float alpha) { ColorFeaturesSegmentator::alpha = alpha; }
    void setTh(float th) { TH = th; }



    // Utility class to make the code more readable
    class Block {
    public:
        /**
         * Constructor of the Block
         * @param m_cb_max Local maximum Cb associated with the block
         * @param m_cr_max Local maximum Cb associated with the block
         * @param m_y_mean Y average of all pixels belonging to the block
         * @param m_start Starting point (top left) of the block in the CbCr plane
         * @param m_end End point (bottom right) of the block in the CbCr plane
         * @param m_pixelCoords Vector containing all pixel coordinates in the original image belonging to the block
         * @param m_label Block label: one of the four main colours
         */
        Block(int m_cb_max, int m_cr_max, int m_y_mean, cv::Point2i m_start, cv::Point2i m_end, std::vector<cv::Point2i> m_pixelCoords, cv::Vec3b m_label) :
            cb_max{m_cb_max}, cr_max{m_cr_max}, y_mean{m_y_mean}, start{m_start}, end{m_end}, pixelsCoords{m_pixelCoords}, label{m_label} {}

        // Getters
        int getCbMax() const {return cb_max;}
        int getCrMax() const {return cr_max;}
        int getYmean() const {return y_mean;}
        cv::Point2i getStart() const {return start;}
        cv::Point2i getEnd() const {return end;}
        std::vector<cv::Point2i> getPixelsCoords() const {return pixelsCoords;}
        cv::Vec3b getLabel() const {return label;}

    private:
        // All pixel coordinates in the original image belonging to the block
        std::vector<cv::Point2i> pixelsCoords;

        // Top left and bottom right points in the CbCr plane (block boundaries)
        cv::Point2i start, end;

        // (Cb, Cr, Ymean) that identifies the block in the CbCr plane, plus the Y average of the associated pixels
        int cb_max;
        int cr_max;
        int y_mean;

        // Label of the block
        cv::Vec3b label;
    };



    // Utility class to make the code more readable
    class Cluster {
    public:
        /**
         * Constructor of the Cluster
         * @param m_label label of the cluster (one of the four main colors)
         */
        Cluster(cv::Vec3b m_label) : label{m_label} {}

        // Getters
        cv::Vec3b getLabel() const { return label; }
        std::vector<cv::Point2i> getPixelsCoords() const { return pixelsCoords; }

        /**
         * Adds a point to the cluster
         * @param p A point containing the coordinates of a pixel
         */
        void addPixel(cv::Point2i p) {pixelsCoords.push_back(p);}

    private:
        // Label of the cluster
        cv::Vec3b label;

        // Vector of points belonging to the cluster
        std::vector<cv::Point2i> pixelsCoords;
    };


private:
    // Member variables
    float N;        // kernel parameter
    float alpha;    // kernel parameter
    float TH;       // the gray color threshold value

    // ********** PRIVATE METHODS **********
    /**
     * Given a YCbCr image returns its normalised histogram by storing it in the input variable 'histogram'.
     * @param yCbCrImage A YCbCr image
     * @param histogram The resulting histogram
     */
    void getNormalizedCbCrHistogram(cv::Mat yCbCrImage, cv::Mat& histogram) const;

    /**
     * Given a histogram, it estimates the normalised PDF through convolution with a Gaussian kernel (using the segmentator's
     * internal parameters) by saving it in the input variable "estimated2dPDF".
     * @param hist A 2D histogram
     * @param estimated2dPDF The resulting estimated normalised PDF
     */
    void estimated2Dpdf(cv::Mat hist, cv::Mat& estimated2dPDF) const;

    /**
     * Given an estimated normalised PDF computes the two independent PDFs of Cr and Cb saving them in the input vectors "pdfCbArray" and "pdfCrArray".
     * @param estimated2dPDF An estimated normalised PDF
     * @param pdfCbArray The resulting independent PDF of Cb
     * @param pdfCrArray The resulting independent PDF of Cr
     */
    void calculateIndependentPDFs(cv::Mat estimated2dPDF, std::vector<float>& pdfCbArray, std::vector<float>& pdfCrArray) const;

    /**
     * Given a 1D pdf, it calculates the indexes of the local maxima and minima by saving them in the input vectors "mn" and "mx".
     * @param arr Input 1D pdf
     * @param mn The vector containing the indices of local minima
     * @param mx The vector containing the indices of local maxima
     */
    void findLocalMaximaMinima(std::vector<float> arr, std::vector<int>& mn, std::vector<int>& mx) const;

    /**
     * Given vectors containing the indices of the local minima of the two independent PDFs, it inserts into
     * the input vector "allPoints" the coordinates of the combinations of these in the CbCr plane that represents the
     * boundaries of the blocks.
     * @param indexMinCb The vector containing the indices of the local minima of the Cb PDF
     * @param indexMinCr The vector containing the indices of the local minima of the Cr PDF
     * @param allPoints Points representing the boundaries of the block in the CbCr plane, they are saved as the starting
     *                  point (top left) and immediately following the end point of the block (bottom right).
     */
    void generateStartEndBlocksPoint(std::vector<int> indexMinCb, std::vector<int> indexMinCr, std::vector<cv::Point2i>& allPoints) const;

    /**
     * Main function that creates all blocks.
     *
     * @param allPoints Vector that contains all "boundaries points" (top left and bottom right) of all blocks
     * @param indexMaxCb The vector containing the indices of local maxima
     * @param indexMaxCr The vector containing the indices of local maxima
     * @param yCrCbImage A YCbCr image
     * @param allBlocks The resulting vector containing all blocks with their characteristics
     */
    void computeAllBlocks(std::vector<cv::Point2i> allPoints, std::vector<int> indexMaxCb, std::vector<int> indexMaxCr, cv::Mat yCrCbImage, std::vector<Block>& allBlocks) const;

    /**
     * Given the start and end points (top left and bottom right) of a block, it returns the average Y value of all
     * pixels belonging to that block and fills the input vector "pixelCoords" with their coordinates relative to the original image.
     *
     * @param yCrCbImage A YCbCr image
     * @param start The top left point of the block in the CbCr plane
     * @param end The bottom right point of the block in the CbCr plane
     * @param pixelCoords The vector to be filled with the coordinates (in the original image) of the pixels belonging to the block
     * @return The average Y of all pixels belonging to the block
     */
    int computeYmean(cv::Mat yCrCbImage, cv::Point2i start, cv::Point2i end, std::vector<cv::Point2i>& pixelCoords) const;

    /**
     * Computes the label to be assigned to the block from those contained in PRINCIPAL_COLORS_RGB vector
     * @param start The top left point of the block in the CbCr plane
     * @param end The bottom right point of the block in the CbCr plane
     * @return The label to be assigned to the block (one of those contained in the PRINCIPAL_COLORS_RGB vector)
     */
    cv::Vec3b computeBlockLabel(cv::Point2i start, cv::Point2i end) const;

    /**
     * It puts together all the blocks with the same label in the same cluster to then generate the segmented image
     * @param allBlocks All computed blocks of image in CbCr plane
     * @param finalClusters The resulting vector of Cluster ("merging" of the previous blocks) that will be filled
     */
    void computeFinalClusters(std::vector<Block> allBlocks, std::vector<Cluster>& finalClusters) const;

    /**
     * Given various clusters fills the input image ("segmentedImage") with the colour of the associated label
     * @param finalClusters The vector that contains all clusters of the four possible label
     * @param segmentedImage The result of the segmentation
     */
    void fillSegmentedImage(std::vector<Cluster> finalClusters, cv::Mat& segmentedImage) const;

    /**
     * Given a segmented image, merge the small regions with the larger surrounding region by changing the labels
     * @param segmentedImage The resulting image with the small regions merged with the larger neighbouring regions
     */
    void mergeSmallRegions(cv::Mat& segmentedImage) const;



    // ********** UTILS METHODS **********
    /**
     * Creates a window and displays the input image with its name and waits for a key to be pressed to continue execution
     * @param img The input image to show
     * @param name The name of the displayed window
     */
    void showImage(cv::Mat img, std::string name) const;

    /**
     * Given a vector and a name, draw the histogram that has in the x-axis the index of the vector and in the y-axis
     * the value at that index in the vector.
     * @param vec A vector of floating point values
     * @param name The name of the displayed window
     */
    void displayHistogram(std::vector<float> vec, std::string name) const;

    /**
     * Like the "displayHistogram" method, but marks the maxima in red and the minima in blue.
     * @param vec A vector of floating point values
     * @param indexMin The vector containing the indices of the local minima of the input "vec"
     * @param indexMax The vector containing the indices of the local maxima of the input "vec"
     * @param name The name of the displayed window
     */
    void displayHistogramMinMax(std::vector<float> vec, std::vector<int> indexMin, std::vector<int> indexMax, std::string name) const;

    /**
     * Draw and display in the CbCr plane the block boundaries with their maxima in red and the points (top left and bottom right) in green
     * @param CbCrHistogram A 2D CbCr histogram
     * @param allPoints Vector that contains all "boundaries points" (top left and bottom right) of all blocks
     * @param indexMinCb The vector containing the indices of the local minima of the Cb PDF
     * @param indexMaxCb The vector containing the indices of the local maxima of the Cb PDF
     * @param indexMinCr The vector containing the indices of the local minima of the Cr PDF
     * @param indexMaxCr The vector containing the indices of the local maxima of the Cr PDF
     */
    void displayBlocksBoundaries(cv::Mat CbCrHistogram, std::vector<cv::Point2i> allPoints, std::vector<int> indexMinCb, std::vector<int> indexMaxCb, std::vector<int> indexMinCr, std::vector<int> indexMaxCr) const;

     /**
      * Draw and display in the CbCr plane the blocks with their colours and maxima
      * @param CbCrHistogram A 2D CbCr histogram
      * @param allBlocks Vector that contains all "boundaries points" (top left and bottom right) of all blocks
      */
    void displayBlocks(cv::Mat CbCrHistogram, std::vector<Block> allBlocks) const;

    /**
     * Reverse mapping according to blocks and their labels
     * @param bgrImage The original image
     * @param allBlocks All computed blocks of image in CbCr plane
     */
    void displayRequantizedImage(cv::Mat bgrImage, std::vector<Block> allBlocks) const;

    /**
     * Convert matrix type to string
     * @param type Integer that represents the type of the matrix
     * @return string type
     */
    std::string type2str(int type) const;
};


#endif //SPORTVIDEOANALYSIS_SADVISION_COLORFEATURESSEGMENTATOR_H
