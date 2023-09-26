#pragma once
#ifndef _SAM_H_
#define _SAM_H_

#include <opencv2/opencv.hpp>
#include <vector>

cv::Size get_preprocess_shape(int oldh, int oldw, int long_side_length);

cv::Mat apply_image(const cv::Mat& image, const int& image_size);

std::vector<float> encode_img(const char* img_path, const wchar_t* img_encoder_path);

std::vector<float> inference(std::vector<float> coords1, std::vector<float> coords2, std::vector<float> output,
    std::vector<float> image_embedding, const wchar_t* onnx_model_path, cv::Mat original_image);

extern "C" __declspec(dllexport) int encode_img_sharp(
    const char* img_path, 
    const char* img_encoder_path, 
    float* outputArray, 
    int* outputSize);

extern "C" __declspec(dllexport) int inference_sharp(
    const float* coords1, int coords1_size,
    const float* coords2, int coords2_size,
    const float* image_embedding, int image_embedding_size,
    const char* onnx_model_path, const char* img_path,
    float* output, int* output_size);

#endif