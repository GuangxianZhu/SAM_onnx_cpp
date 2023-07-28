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
    // std::cout << "apply_image.resized_image.size: " << resized_image.size() << std::endl; //[1024 x 683]
    
    // Normalization per channel, pixel_mean({ 123.675, 116.28, 103.53 }), pixel_std({ 58.395, 57.12, 57.375 })
    cv::Mat channels[3];
    cv::split(resized_image, channels);
    channels[0] = (channels[0] - 123.675) / 58.395;
    channels[1] = (channels[1] - 116.28) / 57.12;
    channels[2] = (channels[2] - 103.53) / 57.375;
    cv::merge(channels, 3, resized_image);
    //debug: convert to vector, and print first 30 elements of vector.
    //std::vector<float> resized_image_vec;
    //resized_image_vec.assign((float*)resized_image.data, (float*)resized_image.data + resized_image.total() * resized_image.channels());
    //std::cout << "apply_image.after_norm" << std::endl;
    //for (int i = 0; i < 30; i++) {
	//	std::cout << resized_image_vec[i] << " ";
	//}

    
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
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, x_img_vec.data(), x_img_vec.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), outputShape.data(), outputShape.size());

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

    return output;
}

cv::Mat SAMImgEncoder::inference(std::vector<cv::Point2f> coords, std::vector<float> image_embedding, cv::Mat img, cv::Mat original_image) {

    // prepare input tensors: image_embedding, onnx_coord, onnx_label, onnx_mask_input, onnx_has_mask_input, orig_im_size
    const std::array<int64_t, 4> image_embedding_shape = { 1, 256, 64, 64 };

    const std::array<int64_t, 3> onnx_coord_shape = { 1, 2, 2 };
    // 1024/img.cols * coordx, 1024/img.rows * coordy
    //std::vector<float> onnx_coord = { 284.44, 213.43, 0.0, 0.0 };//500*0.56=284.44, 375*0.56=213.43. which 1024/1800=0.56; 0,0 is 2nd coord
    //std::vector<float> onnx_coord = { 455.8, 143.1, 0.0, 0.0 }; // motor.jpg
    float coord_x = 1024.0f / static_cast<float>(original_image.cols) * coords[0].x;
    float coord_y = 1024.0f / static_cast<float>(original_image.cols) * coords[0].y;
    std::vector<float> onnx_coord = { coord_x, coord_y, 0.0, 0.0 }; // dog.jpg

    const std::array<int64_t, 2> onnx_label_shape = { 1, 2 };
    std::vector<float> onnx_label = { 1.0, -1.0 };

    const std::array<int64_t, 4> onnx_mask_input_shape = { 1, 1, 256, 256 };
    std::vector<float> onnx_mask_input(1 * 1 * 256 * 256, 0.0f);

    const std::array<int64_t, 1> has_mask_input_shape = { 1 };
    std::vector<float> has_mask_input(1, 0.0f);

    const std::array<int64_t, 1> orig_im_size_shape = { 2 };
    std::vector<float> orig_im_size = { static_cast<float>(img.rows), static_cast<float>(img.cols) };


    // set session
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session(nullptr);
    auto onnx_model_path = L"F:\\Hacarus\\SAM_zhu\\savedmodel\\sam_onnx_example.onnx";
    session = Ort::Session(env, onnx_model_path, Ort::SessionOptions{ nullptr });

    // get input names
    size_t num_inputs = session.GetInputCount();
    std::cout << "Number of inputs: " << num_inputs << std::endl;

    // Create a vector to hold the input tensors
    std::vector<Ort::Value> inputTensors;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, image_embedding.data(), image_embedding.size(), image_embedding_shape.data(), image_embedding_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, onnx_coord.data(), onnx_coord.size(), onnx_coord_shape.data(), onnx_coord_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, onnx_label.data(), onnx_label.size(), onnx_label_shape.data(), onnx_label_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, onnx_mask_input.data(), onnx_mask_input.size(), onnx_mask_input_shape.data(), onnx_mask_input_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, has_mask_input.data(), has_mask_input.size(), has_mask_input_shape.data(), has_mask_input_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, orig_im_size.data(), orig_im_size.size(), orig_im_size_shape.data(), orig_im_size_shape.size()));

    int pixels = img.rows * img.cols;
    const std::array<int64_t, 4> masksShape = { 1, 1, img.rows, img.cols };
    std::vector<float> output(pixels);
    const std::array<int64_t, 2> scoreShape = { 1, 1 };
    std::vector<float> score(1);
    const std::array<int64_t, 4> low_res_logits = { 1, 1, 256, 256 };
    std::vector<float> low_res_logits_output(1 * 1 * 256 * 256);


    std::vector<Ort::Value> outputTensors;
    outputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), masksShape.data(), masksShape.size()));
    outputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, score.data(), score.size(), scoreShape.data(), scoreShape.size()));
    outputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, low_res_logits_output.data(), low_res_logits_output.size(), low_res_logits.data(), low_res_logits.size()));


    Ort::AllocatorWithDefaultOptions ort_alloc;
    // set inputnames as a vector, length = 6
    Ort::AllocatedStringPtr inputName0 = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr inputName1 = session.GetInputNameAllocated(1, ort_alloc);
    Ort::AllocatedStringPtr inputName2 = session.GetInputNameAllocated(2, ort_alloc);
    Ort::AllocatedStringPtr inputName3 = session.GetInputNameAllocated(3, ort_alloc);
    Ort::AllocatedStringPtr inputName4 = session.GetInputNameAllocated(4, ort_alloc);
    Ort::AllocatedStringPtr inputName5 = session.GetInputNameAllocated(5, ort_alloc);
    const std::array<const char*, 6> inputNames = { inputName0.get(), inputName1.get(), inputName2.get(), inputName3.get(), inputName4.get(), inputName5.get() };
    Ort::AllocatedStringPtr outputName0 = session.GetOutputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName1 = session.GetOutputNameAllocated(1, ort_alloc);
    Ort::AllocatedStringPtr outputName2 = session.GetOutputNameAllocated(2, ort_alloc);
    const std::array<const char*, 3> outputNames = { outputName0.get() , outputName1.get(), outputName2.get() };

    // run
    Ort::RunOptions runOptions;
    session.Run(runOptions, inputNames.data(), inputTensors.data(), 6, outputNames.data(), outputTensors.data(), 3);

    // convert to cv::Mat
    cv::Mat output_mask = cv::Mat(img.rows, img.cols, CV_32FC1, output.data());
    output_mask.convertTo(output_mask, CV_8UC1);

    // show mask overlapped on image
    // Normalize the mask to range between 0 and 255
    cv::normalize(output_mask, output_mask, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Resize the mask to the size of original image
    cv::resize(output_mask, output_mask, original_image.size());

    // Make sure both images are of the same type
    original_image.convertTo(original_image, CV_8UC1);

    // 1. Threshold the mask
    float thresh = 0.5;
    cv::threshold(output_mask, output_mask, 0, 255, cv::THRESH_BINARY);

    // 2. Create a colored mask
    cv::Mat colored_mask = cv::Mat::zeros(original_image.size(), CV_8UC3);
    // Here, I'm assuming that the color you want is light blue [255, 229, 204]
    colored_mask.setTo(cv::Scalar(255, 229, 204), output_mask); // BGR color

    // 3. Overlay the mask
    cv::Mat result;
    cv::addWeighted(original_image, 0.5, colored_mask, 0.5, 0.0, result);

    return result;

}

