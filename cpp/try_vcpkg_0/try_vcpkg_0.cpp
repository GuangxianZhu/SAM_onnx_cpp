// try_vcpkg_0.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <commdlg.h>
#include "SAMImgEncoder.h"


std::vector<cv::Point2f> coords;  // Global vector to store clicked points
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        coords.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
}

int main() {

    // load image
    cv::String filepath = "F:\\Hacarus\\SAM_zhu\\notebooks\\images\\motor.jpg";
    cv::Mat img = cv::imread(filepath);
    if (img.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // to float
    img.convertTo(img, CV_32FC3, 1.0f);

    // encode image
    std::cout << "plwase wait about 40 sec ..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    SAMImgEncoder sam_img_encoder;
    std::vector<float> image_embedding = sam_img_encoder.encode_img(img);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "encode_img duration (sec): " << duration.count() / 1000.0 << std::endl;


    // let user click on original_image to get coord
    cv::Mat original_image = cv::imread(filepath);

    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Image", CallBackFunc);

    char key;
    do
    {
        // Show the image
        cv::imshow("Image", original_image);

        if (!coords.empty())
        {
            cv::Mat result = sam_img_encoder.inference(coords, image_embedding, img, original_image);
            cv::imshow("result", result);

            coords.clear();  // clear the vector for next click
        }

        key = cv::waitKey(0);  // wait indefinitely for a keypress
    } while (key != 'q');

    return 0;
}