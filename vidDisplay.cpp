/*
  Yunxuan 'Kaya' Rao
  09/30/2024
  Display live stream video and apply filters on it.
 */
#include <cstdio>  // gives me printf
#include <cstring> // gives me strcpy
#include "opencv2/opencv.hpp" // main OpenCV include file
#include "filters.h"  // Include the filters header
#include "faceDetect.h"

int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window

        // Declare all the variable
        cv::Mat frame;
        cv::Mat grayFrame;
        cv::Mat currFrame;
        cv::Mat altGrayFrame;
        cv::Mat sepiaToneFrame;
        cv::Mat blurredFrame;
        cv::Mat sobelXFrame;
        cv::Mat sobelYFrame;
        cv::Mat newSobelXFrame;
        cv::Mat newSobelYFrame;
        cv::Mat magnitudeFrame;
        cv::Mat blurredQuantizedFrame;
        cv::Mat embossingFrame;
        cv::Mat medianFrame;
        cv::Mat blurOutsideFaceFrame;
        std::vector<cv::Rect> faces;
        int imgCnt = 0;
        bool isGray = false;
        bool isAltGray = false;
        bool isSepiaTone = false;
        bool isFaceDetecting = false;
        bool isBlurred = false;
        bool isXSobel = false;
        bool isYSobel = false;
        bool isMagnitude = false;
        bool isBlurredQuantized = false;
        bool isEmbossing = false;
        bool isMedian = false;
        bool isblurOutsideFace = false;
        
        // Keep the program running until 'q' input
        while (true) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }

                // Switch between different key filters 
                if (isGray){
                    applyGreyscale(frame, grayFrame);
                    currFrame = grayFrame;
                } else if(isAltGray){
                    greyscale(frame, altGrayFrame);
                    currFrame = altGrayFrame;
                }else if (isSepiaTone){
                    applySepiaTone(frame, sepiaToneFrame);
                    currFrame = sepiaToneFrame;
                } else if (isFaceDetecting){
                     // detect faces
                    applyGreyscale(frame, grayFrame);
                    detectFaces( grayFrame, faces );
                    // draw boxes around the faces
                    drawBoxes( frame, faces );
                    currFrame = frame;
                } else if (isBlurred){
                    blur5x5_2(frame, blurredFrame);
                    currFrame = blurredFrame;
                } else if (isXSobel){
                    sobelX3x3(frame, sobelXFrame);
                    cv::convertScaleAbs(sobelXFrame, newSobelXFrame);
                    currFrame = newSobelXFrame;
                } else if (isYSobel){
                    sobelY3x3(frame, sobelYFrame);
                    cv::convertScaleAbs(sobelYFrame, newSobelYFrame);
                    currFrame = newSobelYFrame;
                } else if (isMagnitude){
                    sobelX3x3(frame, sobelXFrame);
                    sobelY3x3(frame, sobelYFrame);
                    magnitude(sobelXFrame, sobelYFrame, magnitudeFrame);
                    currFrame = magnitudeFrame;
                } else if (isBlurredQuantized){
                    blurQuantize(frame, blurredQuantizedFrame, 10);
                    currFrame = blurredQuantizedFrame;
                } else if (isEmbossing){
                    embossingEffect(frame, embossingFrame);
                    currFrame = embossingFrame;
                } else if (isMedian){
                    medianFilter(frame, medianFrame);
                    currFrame = medianFrame;
                } else if (isblurOutsideFace){
                    // detect faces
                    applyGreyscale(frame, grayFrame);
                    detectFaces( grayFrame, faces );
                    blurOutsideFaces(frame, faces, blurOutsideFaceFrame);
                    currFrame = blurOutsideFaceFrame;
                } else {
                    currFrame = frame;
                }     
                cv::imshow("Video", currFrame);          
                
                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                if( key == 'q') {
                    break;
                }

                if (key == 's'){
                    std::string imgname = "captured_video_img" + std::to_string(imgCnt) + ".jpg";
                    cv::imwrite(imgname, currFrame);
                    imgCnt += 1;
                }

                if (key == 'g'){
                    // display greyscale version of video
                    isGray = !isGray;
                }

                if (key == 'h'){
                    // display alt greyscale version of video
                    isAltGray = !isAltGray;
                }

                if (key == 'j'){
                    // display sepia tone version of video
                    isSepiaTone = !isSepiaTone;
                }

                if(key == 'f'){
                    // display face detecting
                    isFaceDetecting = !isFaceDetecting;
                }
                if(key == 'b'){
                    // display alternate blurred version of video
                    isBlurred = !isBlurred;
                }
                if (key == 'x'){
                    // display sobelX version of video
                    isXSobel = !isXSobel;
                }
                if (key == 'y'){
                    // display sobelY version of video
                    isYSobel = !isYSobel;
                }
                if (key == 'm'){
                    // display magnitude image
                    isMagnitude = !isMagnitude;
                }

                if (key == 'l'){
                    // display blurred and quantized color image
                    isBlurredQuantized = !isBlurredQuantized;
                }


                if(key == 'e'){
                    // display embossing effect version of video
                    isEmbossing = !isEmbossing;
                }
                if (key == 'a'){
                    // display median filter version of video
                    isMedian = !isMedian;
                }
                if (key == 'd'){
                    // display blur outside face of video
                    isblurOutsideFace = !isblurOutsideFace;
                }


        }

        delete capdev;
        cv::destroyAllWindows();
        return(0);
}
