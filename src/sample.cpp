#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

class classifier {
public:
    classifier( double TH_ ) : TH( TH_ ) {
        std::cout << "Init classifier\n";
        //sft_ptr = cv::xfeatures2d::SIFT::create( ); 
        /* 0,   // Best features, 0:find all
                                                   5,   // Octave layers (default)
                                                   0.1, // Contrast threshold
                                                   5.,  // Edge threshold
                                                   0.75  // Sigma for Gaussian blur
          );                                            // Create a SIFT detector
  */
  	sft_ptr=cv::SIFT::create();
        std::cout << "Threshold: " << TH << std::endl;
    }
    void train( cv::Mat& img ) {
        img_obj = img.clone( );                                       // Store object for later use
        cv::Mat img_gray;                                             // Gray representation of RGB image
        cv::cvtColor( img, img_gray, CV_BGR2GRAY );                   // Convert to mono
        sft_ptr->detect( img_gray, obj_keypoints );                   // Get Features
        sft_ptr->compute( img_gray, obj_keypoints, obj_descriptors ); // Get Descriptors
        std::cout << "Got a [" << obj_descriptors.rows << "x" << obj_descriptors.cols << "] descriptor matrix\n";
    }
    void match( cv::Mat& img, cv::Mat& result, int min_MATCH = 5 ) {
        std::vector< cv::DMatch > matches;                    // Memory for matches
        std::vector< cv::KeyPoint > keypoints;                // Memory for Keypoints
        cv::Mat descriptors;                                  // Memory for Keypoints
        cv::Mat img_gray;                                     // Gray representation of RGB image
        cv::cvtColor( img, img_gray, CV_BGR2GRAY );           // Convert to mono
        sft_ptr->detect( img_gray, keypoints );               // Get Features
        sft_ptr->compute( img_gray, keypoints, descriptors ); // Get Descriptors
        //--- Match with learned object ---//
        cv::BFMatcher matcher( cv::NORM_L2 ); // Create BF matcher. This datastructure contains info for math descriptors and metrices
        matcher.match( obj_descriptors, descriptors, matches ); // Do matching

        std::cout << "Got " << matches.size( ) << " matches\nFilter matches...\n";
        std::vector< cv::DMatch > goodMatches; // Memory for matches
        for( size_t i = 0; i < matches.size( ); i++ ) {
            if( matches[ i ].distance < TH ) {
                goodMatches.push_back( matches[ i ] );
                // goodKeypoints.push_back(keypoints[matches[i].imgIdx]);
            }
        }
        std::cout << "Result: " << goodMatches.size( ) << " matches\n";
        // std::cout << "Result: " << goodKeypoints.size( ) << " matches\n";
        std::cout << "Stored: " << obj_keypoints.size( ) << " matches\n";
        // std::sort(matches.begin(), matches.end());
        // matches.erase(matches.begin() + 10, matches.end());
        if( matches.size( ) > min_MATCH ) {
            try {
                cv::drawMatches( img_obj, obj_keypoints, img, keypoints, goodMatches, result ); // Draw final results
            } catch( ... ) {
                result = img.clone( ); // Dont know... shuld work
            }
        } else {
            result = img.clone( );
        }
    }

private:
    double TH;                                 // Threshold for BFM
    cv::Ptr< cv::SIFT > sft_ptr;  // Sift pointer datastructure
    std::vector< cv::KeyPoint > obj_keypoints; // Memory for Keypoints
    cv::Mat obj_descriptors;                   // Memory for Keypoints
    cv::Mat img_obj;                           // Memory for obj
};
int main( int, char** argv ) {
    double TH = static_cast< double >( std::stoi( argv[ 1 ] ) ); // BFM threshold
    int camID = static_cast< int >( std::stoi( argv[ 2 ] ) );    // Cam ID
    std::cout << "TH:" << TH << std::endl;
    cv::VideoCapture cap( camID ); // Video class
    //--- Set camera resolution ---//
    cap.set( cv::CAP_PROP_FRAME_WIDTH, 1280 );
    cap.set( cv::CAP_PROP_FRAME_HEIGHT, 720 );
    cv::Mat camImg;                                // Image from camera
    cv::namedWindow( "Img", 0 );                   // Create window to plot images
    cv::Rect ROI = cv::Rect( 400, 200, 400, 400 ); // ROI to learn object
    //----------------------------//
    //--- Do object definition ---//
    //----------------------------//
    cv::Mat img_raw; // Image memory for later use
    char key = 0;    // Keyboard val memory
    while( key != 'd' ) {
        cap >> camImg; // Get image
        if( camImg.rows <= 0 ) {
            std::cout << "Cannot use camera stream\n";
            return ( -1 );
        }
        img_raw = camImg.clone( );                                // Get a working copy without drawn ROI
        cv::rectangle( camImg, ROI, cv::Scalar( 0, 255, 0 ), 2 ); // Draw ROI in image
        cv::imshow( "Img", camImg );                              // Move image to window
        key = cv::waitKey( 1 );                                   // Update window
    }
    //------------------//
    //--- Create ROI ---//
    //------------------//
    classifier myClassifier( TH );
    std::cout << "Do detection...\n";
    cv::Mat obj = img_raw( ROI ).clone( ); // Get obj image
    myClassifier.train( obj );             //''Train'' classifier
    cv::Mat result;                        // Matrix for resulting RGB image
    while( key != 27 ) {                   // Do till <ESC> was pressed
        cap >> camImg;                     // Get image from
        if( camImg.rows <= 0 ) {
            std::cout << "Cannot use camera stream\n";
            return ( -1 );
        }
        myClassifier.match( camImg, result, 5 ); // Do SIFT matching
        cv::imshow( "Img", result );             // Move image to window
        key = cv::waitKey( 1 );                  // Update window
    }
    return 0;
}