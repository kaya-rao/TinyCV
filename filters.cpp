/*
  Yunxuan 'Kaya' Rao
  09/30/2024
The filters that's going to apply to the image/live stream
 */
#include "filters.h"
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings

// Implementation of grayscale filter
int applyGreyscale(cv::Mat &src, cv::Mat &dst) {
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    return(0);
}


// Alternative implementation of grayscale filter
int greyscale(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_8UC1); // empty image same size, single channel
    
    for(int i=0; i<src.rows; i++) {
    cv::Vec3b *srcptr = src.ptr<cv::Vec3b>(i);
    uchar *dstptr = dst.ptr<uchar>(i);
    
    // OpenCV by default reads to BGR  B = 0, G = 1, R = 2
    for(int j=0; j < src.cols; j++) {
      dstptr[j] = static_cast<uchar>(
        (std::min({srcptr[j][2], srcptr[j][1], srcptr[j][0]}) 
        + std::max({srcptr[j][2], srcptr[j][1], srcptr[j][0]})) / 2
        );
    }
  }
  return(0);
}


int applySepiaTone(cv::Mat &src, cv::Mat &dst){
    dst.create(src.size(), src.type()); // empty image same size, same type
    
    for(int i=0; i<src.rows; i++) {
    cv::Vec3b *srcptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(i);
    
    // OpenCV by default reads to BGR  B = 0, G = 1, R = 2
    for(int j=0; j < src.cols; j++) {
        // Blue
        dstptr[j][0] = std::min(static_cast<int>(0.272 * srcptr[j][2] + 0.531 * srcptr[j][1] + 0.131 * srcptr[j][0]), 255);
        // Green
        dstptr[j][1] = std::min(static_cast<int>(0.349 * srcptr[j][2] + 0.686 * srcptr[j][1] + 0.168 * srcptr[j][0]), 255);
        // Red
        dstptr[j][2] = std::min(static_cast<int>(0.393 * srcptr[j][2] + 0.769 * srcptr[j][1] + 0.189 * srcptr[j][0]), 255);
    }
  }
  return(0);
}

// use the time library to get the current time
double getTime() {
    struct timeval cur;
    gettimeofday( &cur, NULL );
    return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}

int blur5x5_1( cv::Mat &src, cv::Mat &dst ){
    // Make a copy of the image
    dst.create(src.size(), src.type());
    double start = getTime();
    
    // 5 x 5 blur filter
    int blurFilter[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    int filterSum = 100;
    
    // Apply blur filter
    for(int r = 2; r < src.rows - 2; r++) {
        for(int c = 2; c < src.cols - 2; c++) {

            // The new value for this pixel should be the sum of the 
            // neighbor pixel * weight(filter)
            int newBlue = 0, newGreen = 0, newRed = 0;

            // Calculate the sum
            for(int i = -2; i <= 2; i++){
                for(int j = -2; j <= 2; j++){
                    // Get the value of the pixel
                    cv::Vec3b currPixel = src.at<cv::Vec3b>(r + i, c + j);
                    newBlue += (currPixel[0] * blurFilter[i + 2][j + 2]);
                    newGreen += (currPixel[1] * blurFilter[i + 2][j + 2]);
                    newRed += (currPixel[2] * blurFilter[i + 2][j + 2]);

                }
            }
            dst.at<cv::Vec3b>(r, c)[0] = static_cast<int>(newBlue / filterSum);
            dst.at<cv::Vec3b>(r, c)[1] = static_cast<int>(newGreen / filterSum);
            dst.at<cv::Vec3b>(r, c)[2] = static_cast<int>(newRed / filterSum);
        }
    }
    double end = getTime();
    printf("The original blur filter takes: %.5f\n", end - start);
    return(0);
}



int blur5x5_2( cv::Mat &src, cv::Mat &dst ){
    // Start recording time
    double start = getTime();
    dst.create(src.size(), src.type());

    // Create a temperate image Mat
    cv::Mat tmp = cv::Mat::zeros(src.size(), src.type());
    
    // 1D 5 x 1 blur filter
    int blurFilter[5] = {1, 2, 4, 2, 1};
    int filterSum = 16;
    
    // Horizontal
    for (int r = 0; r < src.rows; r++) {
        for (int c = 2; c < src.cols - 2; c++) {
            int newBlue = 0, newGreen = 0, newRed = 0;

            for (int j = -2; j <= 2; j++) {
                cv::Vec3b srcPixel = src.at<cv::Vec3b>(r, c + j);
                newBlue += srcPixel[0] * blurFilter[j + 2];
                newGreen += srcPixel[1] * blurFilter[j + 2];
                newRed += srcPixel[2] * blurFilter[j + 2];
            }

            tmp.at<cv::Vec3b>(r, c)[0] = static_cast<int>(newBlue / filterSum);
            tmp.at<cv::Vec3b>(r, c)[1] = static_cast<int>(newGreen / filterSum);
            tmp.at<cv::Vec3b>(r, c)[2] = static_cast<int>(newRed / filterSum);
        }
    }

    // Vertical
    for (int r = 2; r < src.rows - 2; r++) {
        for (int c = 0; c < src.cols; c++) {
            int newBlue = 0, newGreen = 0, newRed = 0;

            for (int i = -2; i <= 2; i++) {
                cv::Vec3b tmpPixel = tmp.at<cv::Vec3b>(r + i, c);
                newBlue += tmpPixel[0] * blurFilter[i + 2];
                newGreen += tmpPixel[1] * blurFilter[i + 2];
                newRed += tmpPixel[2] * blurFilter[i + 2];
            }

            dst.at<cv::Vec3b>(r, c)[0] = static_cast<int>(newBlue / filterSum);
            dst.at<cv::Vec3b>(r, c)[1] = static_cast<int>(newGreen / filterSum);
            dst.at<cv::Vec3b>(r, c)[2] = static_cast<int>(newRed / filterSum);
        }
    }

    double end = getTime();
    printf("The alternate blur filter takes: %.5f\n", end - start);
    return(0);
}

// horizontal（X）sobel filter - posotive right
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    // Scales, calculates absolute values, and converts the result to 8-bit.
    // cv::convertScaleAbs(tmp, dst); is moved to vidDisplay.cpp
    // so the magnitude and embossingEffect can use CV_16SC3 format data directly
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    /* SobelX Filter
    int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    Seperable:
    [-1, 0, 1] x [1]
                 [2]
                 [1]
    */
    
    // Horizontal
    for(int r = 0; r < src.rows; r++) {
        for(int c = 1; c < src.cols - 1; c++) {
            for(int i = 0; i < 3; i++){
                temp.at<cv::Vec3s>(r, c)[i] = - src.at<cv::Vec3b>(r, c - 1)[i] + src.at<cv::Vec3b>(r, c + 1)[i];
            }
        }
    }

    // Vertical
    for(int r = 1; r < src.rows - 1; r++) {
        for(int c = 0; c < src.cols; c++) {
            for(int i = 0; i < 3; i++){
                dst.at<cv::Vec3s>(r, c)[i] = temp.at<cv::Vec3s>(r - 1, c)[i] + 2 * temp.at<cv::Vec3s>(r, c)[i] + temp.at<cv::Vec3s>(r + 1, c)[i];;
            }
        }
    }
    return(0);
}

// Vertical(Y) 
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    //cv::Mat tmp = cv::Mat::zeros(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    /* SobelY Filter
    int sobelY[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    
    Seperable:
    [1] x [1, 2, 1] 
    [0]
    [-1]
    };*/

    // Vertical
    for (int r = 1; r < src.rows - 1; r++) {
        for (int c = 0; c < src.cols; c++) {
            for (int i = 0; i < 3; i++) {
                temp.at<cv::Vec3s>(r, c)[i] = src.at<cv::Vec3b>(r - 1, c)[i] - src.at<cv::Vec3b>(r + 1, c)[i];
            }
        }
    }

    // Horizontal
    for (int r = 0; r < src.rows; r++) {
        for (int c = 1; c < src.cols - 1; c++) {  // Avoiding edges
            for (int i = 0; i < 3; i++) {
                dst.at<cv::Vec3s>(r, c)[i] = temp.at<cv::Vec3s>(r, c - 1)[i] + 2 * temp.at<cv::Vec3s>(r, c)[i] + temp.at<cv::Vec3s>(r, c + 1)[i];
            }
        }
    }

    //cv::convertScaleAbs(tmp, dst);
    return(0);
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    dst.create(sx.size(), CV_8UC3); // unsigned char
    // magnitude I = sqrt(sx * sx + sy * sy )
    
    for(int r = 0; r < sx.rows; r++) {
        for(int c = 0; c < sx.cols; c++) {
            cv::Vec3s gradientX = sx.at<cv::Vec3s>(r, c);
            cv::Vec3s gradientY = sy.at<cv::Vec3s>(r, c);
            
            // Calculate the magnitude for R, G, B
            // Filter weight sum up to 0
            for(int i = 0; i < 3; i++){
                int magnitude = static_cast<int>(std::sqrt(gradientX[i] * gradientX[i] + gradientY[i] * gradientY[i]));
                dst.at<cv::Vec3b>(r, c)[i] = static_cast<uchar>(magnitude);
            }
        }
    }
    return(0);
}

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    // Available blur filter: blur5x5_2
    cv::Mat blurredImg;
    cv::GaussianBlur(src, blurredImg, cv::Size(51, 51), 0); 
    // blur5x5_2(src, blurredImg);

    dst.create(src.size(), src.type());

    int bucketSize = 255 / levels; 
    // quantized value = (int(original value / level))* level.

    for (int r = 0; r < blurredImg.rows; r++){
        for( int c = 0; c < blurredImg.cols; c++){
            cv::Vec3b currPixel = blurredImg.at<cv::Vec3b>(r, c);

            for(int i = 0; i < 3; i++){
                int quantizedValue = (currPixel[i] / bucketSize) * bucketSize;
                dst.at<cv::Vec3b>(r, c)[i] = static_cast<uchar>(quantizedValue);
            }
        }
    }
    return(0);
}

// Q11 - 1
int embossingEffect(cv::Mat &src, cv::Mat &dst){
    // Get the sobel output by applying previous filter
    cv::Mat sobelX, sobelY;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);
    dst.create(src.size(), src.type());

    // Directions: 0.7071(Sqrt(2)/2) is 45 degree vector
    float direction = 0.7071;

    for (int r = 0; r < src.rows; r++){
        for(int c = 0; c < src.cols; c++){
            for(int i = 0; i < 3; i++){
                float currPixelX = sobelX.at<cv::Vec3b>(r, c)[i];
                float currPixelY = sobelY.at<cv::Vec3b>(r, c)[i];
                float embossedValue = currPixelX * direction + currPixelY * direction;  // Currently embossedValue is in range [-255 * 2 * 0.7071, 255 * 2 * 0.7071] -> [-360.53, 360.53]

                // Normalize to [0, 255], linear normalization
                embossedValue = (embossedValue + 255.0f) / 2.0f;
                embossedValue = std::min(255.0f, std::max(0.0f, embossedValue));
                dst.at<cv::Vec3b>(r, c)[i] = static_cast<uchar>(embossedValue);
            }
        }
    }
    return(0);
}

//Q11 - 2
// Median filter:
// Find the median value within the neighbors, and put it in the middle.

int medianFilter(cv::Mat &src, cv::Mat &dst){
    dst.create(src.size(), src.type());
    for (int r = 0; r < src.rows; r++){
        for(int c = 0; c < src.cols; c++){
            for(int channel = 0; channel < 3; channel++){
                // implement a 3 x 3 median filter
                // Collect the value within the box
                std::vector<int> neighborCollection;
                for(int i = -1; i <= 1 ; i++){
                    for(int j = -1; j <= 1 ; j++){
                        neighborCollection.push_back(src.at<cv::Vec3b>(r + i, c + j)[channel]);
                    }
                }
                // Sort and find median
                std::sort(neighborCollection.begin(), neighborCollection.end());
                int medianVal = neighborCollection[neighborCollection.size() / 2];
                dst.at<cv::Vec3b>(r, c)[channel] = medianVal;
            }
        }
    }
    return(0);
}

int blurOutsideFaces(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst){
    // Blur the whole image to dst
    cv::GaussianBlur(src, dst, cv::Size(51, 51), 0); // Apply GaussianBlur with 5 x 5 filter

    // Copy the non-blurred pixel within the faces area from src to dst
    for (size_t i = 0; i < faces.size(); i++){
        src(faces[i]).copyTo(dst(faces[i]));
    }
    return(0);
}
