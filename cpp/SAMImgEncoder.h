#pragma once
#ifndef SAMIMG_ENCODER_H
#define SAMIMG_ENCODER_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 声明类和类的成员函数
class SAMImgEncoder {
public:
    // Constructor
    SAMImgEncoder();

    // ResizeLongestSide: Proportional scaling the image.
    int img_size;
    
    // Quantization
    bool qt;

    // Function to compute new height and width
    cv::Size get_preprocess_shape(int oldh, int oldw, int long_side_length);

    // apply image: 1. resize, 2. normalize, 3. pad to square, 4. HWC to CHW
    cv::Mat apply_image(const cv::Mat& image);

    // encode
    std::vector<float> encode_img(const cv::Mat& image);
};

#endif  // SAMIMG_ENCODER_H


