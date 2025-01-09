/*
  Yunxuan 'Kaya' Rao
  09/22/2024
  Display image and apply filters on images.
 */

#include <cstdio>  // gives me printf
#include <cstring> // gives me strcpy
#include "opencv2/opencv.hpp" // main OpenCV include file
#include "filters.h"  // Include the filters header



int main(int argc, char *argv[]) { // main function, argc == # command line arguments, argv == cmd line strings
  cv::Mat src;  // primary image type
  char filename[256];

  // usage: checking if the user provided a filename
  if(argc < 2) {
    printf("Usage %s <image filename>\n", argv[0]);
    exit(-1);
  }
  strcpy(filename, argv[1]); // copying 2nd command line argument to filename variable

  // read the image
  src = cv::imread( filename );  // imread by default converts all inputs to 8 bit unsigned RGB or greyscale
  if( src.data == NULL ) {
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }
  cv::imshow( "Original Image", src );

  // Use two 5 X 5 blur filter and count the time it takes
  cv::Mat blur5x5_1Img, blur5x5_2Img;
  blur5x5_1(src, blur5x5_1Img);
  blur5x5_2(src, blur5x5_2Img);
  cv::imshow( "Original Blur", blur5x5_1Img );
  cv::imshow( "Alt Blur", blur5x5_2Img );

  // Use embossing effect and median filter and display the image
  cv::Mat embossingFrame;
  cv::Mat medianFrame;
  embossingEffect(src, embossingFrame);
  medianFilter(src, medianFrame);
  cv::imshow( "Embossing", embossingFrame );
  cv::imshow( "Median", medianFrame );


  while(true){
    char key = (char)cv::waitKey(1);
    if(key == 'q'){
      break;
    }
  }
  
  cv::waitKey(0);
  cv::destroyWindow( "Original Image");

  return(0);
}
  

  
  

