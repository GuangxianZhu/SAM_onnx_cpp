#include "SAMImgEncoder.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <array>

SAMImgEncoder::SAMImgEncoder() :
	img_size(1024), qt(false)
{
}

cv::Size SAMImgEncoder::get_preprocess_shape(int oldh, int oldw, int long_side_length) {
    float scale = static_cast<float>(long_side_length) / std::max(oldh, oldw);
    int newh = static_cast<int>(oldh * scale + 0.5);
    int neww = static_cast<int>(oldw * scale + 0.5);
    return cv::Size(neww, newh);
}

cv::Mat SAMImgEncoder::apply_image(const cv::Mat& image) {

    cv::Mat resized_image, padded_image, chw_image;

    cv::Size target_size = get_preprocess_shape(image.rows, image.cols, img_size);
    //std::cout << "apply_image.target_size: " << target_size << std::endl; // w:1024, h:683
    
    // resize, interpolation=cv::INTER_LINEAR
    cv::resize(image, resized_image, target_size, 0, 0, cv::INTER_LINEAR);
    // debug:print resized_image size
    std::cout << "apply_image.resized_image.size: " << resized_image.size() << std::endl; //[1024 x 683]
    
    // Normalization per channel, pixel_mean({ 123.675, 116.28, 103.53 }), pixel_std({ 58.395, 57.12, 57.375 })
    cv::Mat channels[3];
    cv::split(resized_image, channels);
    channels[0] = (channels[0] - 123.675) / 58.395;
    channels[1] = (channels[1] - 116.28) / 57.12;
    channels[2] = (channels[2] - 103.53) / 57.375;
    cv::merge(channels, 3, resized_image);
    //debug: convert to vector, and print first 30 elements of vector.
    std::vector<float> resized_image_vec;
    resized_image_vec.assign((float*)resized_image.data, (float*)resized_image.data + resized_image.total() * resized_image.channels());
    std::cout << "apply_image.after_norm" << std::endl;
    for (int i = 0; i < 30; i++) {
		std::cout << resized_image_vec[i] << " ";
	}

    
    //pad to square
    int h = resized_image.rows;
    int w = resized_image.cols;
    int pad_h = img_size - h;
    int pad_w = img_size - w;
    assert(pad_h >= 0 && pad_w >= 0);
    cv::copyMakeBorder(resized_image, padded_image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    //HWC to CHW
    cv::dnn::blobFromImage(padded_image, chw_image);

    return chw_image;

}


std::vector<float> SAMImgEncoder::encode_img(const cv::Mat& x) {
	// apply image: 1. resize, 2. normalize, 3. pad to square, 4. HWC to CHW
    cv::Mat x_img = apply_image(x);
    
    // copy x_img to vector
    std::vector<float> x_img_vec;
    x_img_vec.assign((float*)x_img.data, (float*)x_img.data + x_img.total() * x_img.channels());

    // set input, output array
    constexpr int64_t numInputElements = 3 * 1024 * 1024;
    constexpr int64_t numOutputElements = 256 * 64 * 64;
    std::vector<float> input(numInputElements);
    std::vector<float> output(numOutputElements);

    // set shape
    const std::array<int64_t, 4> inputShape = { 1, 3, 1024, 1024 };
    const std::array<int64_t, 4> outputShape = { 1, 256, 64, 64 };
	
    // paths
    auto img_encoder_path = L"F:\\Hacarus\\SAM_zhu\\savedmodel\\ImageEncoderViT.onnx";
    auto img_encoder_path_qt = L"F:\\Hacarus\\SAM_zhu\\savedmodel\\ImageEncoderViT_qt.onnx";

    Ort::Env env; // create environment
    Ort::RunOptions runOptions; // set run options if needed
    Ort::Session session(nullptr); //

    // set sesson
    Ort::SessionOptions ort_session_options;
    session = Ort::Session(env, img_encoder_path, Ort::SessionOptions{ nullptr });

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), outputShape.data(), outputShape.size());

    // copy image data to input array
    std::copy(x_img_vec.begin(), x_img_vec.end(), input.begin());


    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };
    inputName.release();
    outputName.release();

    // run
    session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

    // copy output array to vector
    std::vector<float> output_vec;
    for (int i = 0; i < output.size(); ++i) {
        output_vec.push_back(output[i]);
    }

    return output_vec;
}

