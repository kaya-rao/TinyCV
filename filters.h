/*
  Yunxuan 'Kaya' Rao
  09/30/2024
The header of the filters that's going to apply to the image/live stream
 */
#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// Function prototypes for filters

// Grayscale with cvtColor
int applyGreyscale(cv::Mat &src, cv::Mat &dst);

// Alternative greyscale 
int greyscale(cv::Mat &src, cv::Mat &dst);

// Sepia Tone
int applySepiaTone(cv::Mat &src, cv::Mat &dst);

// 5 X 5 blur filter
int blur5x5_1( cv::Mat &src, cv::Mat &dst );

// 2 5 X 1 filter implement as a 5 X 5 blur filter
int blur5x5_2( cv::Mat &src, cv::Mat &dst );

// 3x3 Sobel X filter as separable 1x3 filters
int sobelX3x3( cv::Mat &src, cv::Mat &dst );

// 3x3 Sobel Y filter as separable 1x3 filters
int sobelY3x3( cv::Mat &src, cv::Mat &dst );


// Generates a gradient magnitude image from the X and Y Sobel images
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

// Blurs and quantizes a color image
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

// Q10 is linked at vidDisplay.cpp

// Q11
// Embossing Effect
int embossingEffect(cv::Mat &src, cv::Mat &dst);

// Median Filter
int medianFilter(cv::Mat &src, cv::Mat &dst);

// Blur Outside of Faces
int blurOutsideFaces(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst);


#endif // FILTERS_H
