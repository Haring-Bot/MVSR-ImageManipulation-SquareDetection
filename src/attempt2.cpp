#include <iostream>
#include "opencv2/opencv.hpp"

//class for most minor tools
class toolbox
{
private:
    //init values for HSV segmentation
    int lH = 47;
    int uH = 60;
    int lS = 45;
    int uS = 124;
    int lV = 158;
    int uV = 255;
    int lH2 = lH;   //secondary H values
    int uH2 = uH;

    std::string lastPath;
    cv::Mat originalImage;

    //callback udpates segmentation if slider has changed
    static void updateSlider(int, void *userdata)   
    {
        toolbox *seg = reinterpret_cast<toolbox *>(userdata);
        seg->segment();
    }

public:
    toolbox()
    {
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    }

cv::Mat segment(std::string path)                                       //segments BGR picture in the HSV spectrum
    {
        originalImage = cv::imread(path);                               //ensures picture is readable
        if (originalImage.empty())
        {
            std::cout << "no picture found" << std::endl;
            return cv::Mat();
        }

        cv::Mat hsv_image;
        cv::cvtColor(originalImage, hsv_image, cv::COLOR_BGR2HSV);        //converts image from BGR to HSV

        if (lH > uH)                                                      //this allows the segmenter to segment H values between a 179 and 0 degrees
        {                                                                 //especially useful since red is aprx. 170-10
            lH2 = lH;
            uH2 = 179;
            lH = 0;
            // std::cout << "value changed \n";
        }
        else
        {
            lH2 = lH;
            uH2 = uH;
        }

        cv::Mat hsvMask1, hsvMask2;
        cv::Scalar lowerLimit1(lH, lS, lV);                                 
        cv::Scalar upperLimit1(uH, uS, uV);
        cv::inRange(hsv_image, lowerLimit1, upperLimit1, hsvMask1);         //extracts all values within limits into hsvMask1.

        cv::Scalar lowerLimit2(lH2, lS, lV);
        cv::Scalar upperLimit2(uH2, uS, uV);
        cv::inRange(hsv_image, lowerLimit2, upperLimit2, hsvMask2);

        cv::Mat combinedMask;
        cv::bitwise_or(hsvMask1, hsvMask2, combinedMask);                   //combines extracted values from mask1 and 2

        cv::Mat segmentedImage;
        cv::bitwise_and(originalImage, originalImage, segmentedImage, combinedMask);    //applies mask to original picture keeping only the values
                                                                                        //present in the mask and the picture
        cv::imshow("before", originalImage);
        cv::imshow("after", segmentedImage);

        //sliders for segmentation
        cv::createTrackbar("lowerHue", "after", &lH, 179, updateSlider, this);
        cv::createTrackbar("upperHue", "after", &uH, 179, updateSlider, this);
        cv::createTrackbar("lowerSaturation", "after", &lS, 255, updateSlider, this);
        cv::createTrackbar("upperSaturation", "after", &uS, 255, updateSlider, this);
        cv::createTrackbar("lowerValue", "after", &lV, 255, updateSlider, this);
        cv::createTrackbar("upperValue", "after", &uV, 255, updateSlider, this);

        lastPath = path;                                                        //saves path used in this instance for empty function call

        return segmentedImage;
    }

    cv::Mat segment()
    {
        return segment(lastPath);                                                //if the function is called empty (as done by the callback of the slider)
    }                                                                            //the path of the last picture is continued to be used

    cv::Mat decolor(cv::Mat image)                                               //for converting 3-channel images to 1-channel grey-scale pictures
    {
        cv::Mat decoloredImage(image.size(), CV_8UC1);
        for (int y = 0; y < image.rows; y++)                                     //iterates over all rows->columns->channel
        {
            for (int x = 0; x < image.cols; x++)
            {
                uchar H = image.at<cv::Vec3b>(y, x)[0];
                uchar S = image.at<cv::Vec3b>(y, x)[1];
                uchar V = image.at<cv::Vec3b>(y, x)[2];

                decoloredImage.at<uchar>(y, x) = (H + S + V) / 3;                  //calculating median of all three chanels and saves at ssame location
            }
        }
        cv::imshow("decolored", decoloredImage);
        return decoloredImage;
    }

    cv::Mat thresholder(cv::Mat image, int threshold)                               //thresholds image between a parameter and 255
    {
        cv::threshold(image, image, threshold, 255, cv::THRESH_BINARY);
        return image;
    }

    cv::Mat combineMat(cv::Mat image1, cv::Mat image2, int visualize)               //for manually combining two images without using the prebuilt function
    {
        std::cout << "trying to combine\n";
        cv::Mat newPicture(image1.size(), CV_8UC1);
        for (int currentX = 0; currentX < image1.rows; currentX++)
        {
            for (int currentY = 0; currentY < image1.cols; currentY++)
            {
                newPicture.at<uchar>(currentX, currentY) = (image1.at<uchar>(currentX, currentY) + image2.at<uchar>(currentX, currentY)) / 2;
            }
        }
        std::cout << "successfully combined\n";

        if (visualize == 1)
        {
            cv::imshow("CombineMat result", newPicture);
        }

        return newPicture;
    }

    cv::Mat otsu(cv::Mat image)                                                     //thresholds between 0 and 255 using OTSU method
    {
        cv::threshold(image, image, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        return image;
    }
};

class discreteConvolution                                                           //for convoluting images
{
private:
    cv::Mat kernel;

public:
    discreteConvolution(){};
    discreteConvolution(cv::Mat kernel) : kernel(kernel) {}

    int kernelMin = -1;
    int kernelMax = 2;

    cv::Mat conv(cv::Mat image, int normalize, int visualisation)
    {
        cv::Mat convolutedImage;
        // cv::Mat convolutedImage = cv::Mat::zeros(image.size(), CV_32F);
        std::vector<cv::Mat> convolutedChannels;
        for (int layer = 0; layer < image.channels(); layer++)                  //iterates over all channels. Allows for the segmentation of n-channel images such as BGR
        {
            cv::Mat currentLayer(image.size(), CV_32FC1);
            cv::Mat convolutedLayer(currentLayer.size(), CV_32FC1);
            cv::extractChannel(image, currentLayer, layer);                     //extracts layer
            for (int y = 0; y < currentLayer.rows; y++)                         //iterates over all rows
            {
                for (int x = 0; x < currentLayer.cols; x++)                     //iterates over all columns
                {
                    float kernelSum = 0;
                    for (int l = kernelMin; l < kernelMax; l++)                 //iterates over all Kernel rows
                    {
                        for (int k = kernelMin; k < kernelMax; k++)             //iterates over all Kernel columns
                        {
                            float pixelVal = static_cast<float>(currentLayer.at<uchar>(y + l, x + k));      //extracts value from original picture at the same position as kernel
                        float kernelValue = kernel.at<float>(l + 1, k + 1);                                 //extracts matching kernel value
                            kernelSum += pixelVal * kernelValue;                                            //calculates product and adds to KernelSum
                        }
                    }
            if (kernelSum < 0)                                                                              //ensures that negative values calculated by sobel are displayed approprioate 
                    {
                        kernelSum = std::abs(kernelSum);
                    }

                convolutedLayer.at<float>(y, x) = kernelSum;                                                //saves layer
                }
            }
        convolutedChannels.push_back(convolutedLayer);                                                      //creates vector with layers for combining later on
        }
        cv::merge(convolutedChannels, convolutedImage);                                                     //combins layers for multi layer convolution

        if (normalize == 1)                                                                                 //nomalize result if requested
        {
            cv::normalize(convolutedImage, convolutedImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);              //result is CV_8UC1
        }

        if (visualisation == 1)                                                                             //visualizes results if requested
        {
            cv::imshow("convoluted image", convolutedImage);
        }
        return convolutedImage;
    }
};

class sobelDetector                                                                                         //detects edges
{
private:
    discreteConvolution ciX;
    discreteConvolution ciY;
    toolbox myToolbox;

public:
    sobelDetector()
    {
        cv::Mat sobelXKernel = (cv::Mat_<float>(3, 3) << -1.0f, 0, 1.0f,                                    //common kernel used for sobel operations
                                -2.0f, 0, 2.0f,
                                -1.0f, 0, 1.0f);

        cv::Mat sobelYKernel = (cv::Mat_<float>(3, 3) << -1.0f, -2.0f, -1.0f,
                                0, 0, 0,
                                1.0f, 2.0f, 1.0f);

        ciX = discreteConvolution(sobelXKernel);                                                            //creates discreteConvolution object
        ciY = discreteConvolution(sobelYKernel);
    }

    cv::Mat getEdges(cv::Mat image, int visualisation)
    {
    cv::Mat sobelImageX = ciX.conv(image, 1, 0);                                                            //calls convolution method
        cv::Mat sobelImageY = ciY.conv(image, 1, 0);

        cv::Mat sobelImageXY = myToolbox.combineMat(sobelImageX, sobelImageY, 0);                           //combines edges on X and Y axis

        sobelImageXY.convertTo(sobelImageXY, CV_8U);                                                        //converts result to CV_8U

    if (visualisation == 1)                                                                                 //visualizes result if requested
        {
            cv::imshow("edges", sobelImageXY);
        }
        return sobelImageXY;
    }
};

class squareDetector                                                                                        //finds circles and displays them
{
public:
    squareDetector(){};
    squareDetector(cv::Mat image, std::string path, int visualisation)
    {
        findSquares(image, path, visualisation);
    }

    cv::Mat findSquares(cv::Mat image, std::string path, int visualisation)
    {
        cv::Mat origPic = cv::imread(path);                                                                 //opens original BGR image
    std::vector<cv::Vec4i> lineVect;                                                                      //defines vector for circle coordinates
    cv::HoughLinesP(image, lineVect, 1, CV_PI/180, 50, 50, 10 );                        //calls hough gradient algorithm for finding
                                                                                                            //cirles giving extracted edges
                                                                                                            //the high minimum distance is because we know we're only looking for one circle
        if (visualisation == 1)
        {
            for (size_t i = 0; i < lineVect.size(); i++)                                                  //for number of circles found
            {
                cv::Vec4i line = lineVect[i];
                cv::line(origPic, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

                std::cout << "Found line between " << line[0] << "/" << line[1] << " and " << line[2] << "/" << line[3] << std::endl;
            }
            cv::imshow("squares", origPic);
        }
        return origPic;
    }
};

int main()
{
    cv::Mat myKernel = (cv::Mat_<float>(3, 3) << 0.0625, 0.125, 0.0625,                                     //gaussian kernel
                                                 0.125, 0.25, 0.125,
                                                0.0625, 0.125, 0.0625);

    toolbox mytoolbox;                                                                                      //creates neccessary objects
    discreteConvolution myDC(myKernel);
    sobelDetector mySobel;
    squareDetector myCircleDetector;
    while (true)
    {
        std::cout << "startup successfull\n";
        cv::Mat myPicture = mytoolbox.segment("../pictures/greenCoco.jpeg");
        std::cout << "segmentation successfull\n";
        myPicture = myDC.conv(myPicture, 1, 1);
        std::cout << "discrete convolution successfull\n";
        myPicture = mytoolbox.decolor(myPicture);
        std::cout << "decoloring successfull\n";
        myPicture = mySobel.getEdges(myPicture, 1);
        std::cout << "getEdge successfull\n";
        // myPicture = mytoolbox.thresholder(myPicture, 170);
        myPicture = mytoolbox.otsu(myPicture);
        cv::imshow("thresholded", myPicture);
        myPicture = myCircleDetector.findSquares(myPicture, "../pictures/greenCoco.jpeg", 1);

        std::cout << "press any key to repeat\n";
cv::waitKey(0);                                                                                             //allows renewal of processes
    }
}
