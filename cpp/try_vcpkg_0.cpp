// try_vcpkg_0.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "SAMImgEncoder.h"

int main() {
    // load image
    cv::Mat img = cv::imread("F:\\Hacarus\\SAM_zhu\\notebooks\\images\\truck.jpg");
    if (img.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // to float
    img.convertTo(img, CV_32FC3, 1.0f);
    //std::cout << "img type: " << img.type() << std::endl; // 5: CV_32FC3

    // test encode_img
    SAMImgEncoder sam_img_encoder;
    std::vector<float> image_embedding = sam_img_encoder.encode_img(img);
    std::cout << "image_embedding size: " << image_embedding.size() << std::endl;
    // debug:print first 30 elements
    for (int i = 0; i < 30; i++) {
		std::cout << image_embedding[i] << " ";
	}
    

    // prepare input tensors: image_embedding, onnx_coord, onnx_label, onnx_mask_input, onnx_has_mask_input, orig_im_size
    // image_embedding is already prepared
    // onnx_coord, python: onnx_coord = [[[284.44 213.43 ] [0.        0.]]], shape: (1, 2, 2)
    // onnx_label, python: onnx_label = [[ 1., -1.]], shape: (1, 2)
    // onnx_mask_input, python: np.zeros((1, 1, 256, 256), dtype=np.float32)
    // has_mask_input, python: np.zeros(1, dtype=np.float32)
    // orig_im_size, python: np.array(img.shape[:2], dtype=np.float32)
    const std::array<int64_t, 4> image_embedding_shape = { 1, 256, 64, 64 };

    const std::array<int64_t, 3> onnx_coord_shape = { 1, 2, 2 };
    // 1024/img.cols * coordx, 1024/img.rows * coordy
    std::vector<float> onnx_coord = { 284.44, 213.43, 630.0, 350.0 };//500*0.56=284.44, 375*0.56=213.43. which 1024/1800=0.56; 0,0 is 2nd coord

    const std::array<int64_t, 2> onnx_label_shape = { 1, 2 };
    std::vector<float> onnx_label = { 1.0, 1.0 };

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

    
    constexpr int64_t numOutputElements = 1 * 1 * 1200 * 1800;
    //const std::array<int64_t, 4> outputShape = { 1, 1, 1200, 1800 };
    const std::array<int64_t, 4> outputShape = { 1, 1, img.rows, img.cols };
    std::vector<float> output(numOutputElements);
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), outputShape.data(), outputShape.size());


    Ort::AllocatorWithDefaultOptions ort_alloc;
    // set inputnames as a vector, length = 6
    Ort::AllocatedStringPtr inputName0 = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr inputName1 = session.GetInputNameAllocated(1, ort_alloc);
    Ort::AllocatedStringPtr inputName2 = session.GetInputNameAllocated(2, ort_alloc);
    Ort::AllocatedStringPtr inputName3 = session.GetInputNameAllocated(3, ort_alloc);
    Ort::AllocatedStringPtr inputName4 = session.GetInputNameAllocated(4, ort_alloc);
    Ort::AllocatedStringPtr inputName5 = session.GetInputNameAllocated(5, ort_alloc);
    const std::array<const char*, 6> inputNames = { inputName0.get(), inputName1.get(), inputName2.get(), inputName3.get(), inputName4.get(), inputName5.get() };
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> outputNames = { outputName.get() };

    // run
    Ort::RunOptions runOptions;
    session.Run(runOptions, inputNames.data(), inputTensors.data(), 6, outputNames.data(), &outputTensor, 1);
    
    // copy output to vector
    std::vector<float> output_vec;
    for (int i = 0; i < output.size(); ++i) {
        output_vec.push_back(output[i]);
    }
    
    // convert to cv::Mat
    cv::Mat output_mask = cv::Mat(1200, 1800, CV_32FC1, output_vec.data());
    output_mask.convertTo(output_mask, CV_8UC1);

    cv::Mat original_image = cv::imread("F:\\Hacarus\\SAM_zhu\\notebooks\\images\\truck.jpg");

    // show mask overlapped on image
    // Normalize the mask to range between 0 and 255
    cv::normalize(output_mask, output_mask, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    // Resize the mask to the size of original image
    cv::resize(output_mask, output_mask, original_image.size());

    // Make sure both images are of the same type
    original_image.convertTo(original_image, CV_8UC1);

    // 1. Threshold the mask
    cv::threshold(output_mask, output_mask, 0, 255, cv::THRESH_BINARY);

    // 2. Create a colored mask
    cv::Mat colored_mask = cv::Mat::zeros(original_image.size(), CV_8UC3);
    // Here, I'm assuming that the color you want is light blue [255, 229, 204]
    colored_mask.setTo(cv::Scalar(255, 229, 204), output_mask); // BGR color

    // 3. Overlay the mask
    cv::Mat result;
    cv::addWeighted(original_image, 0.5, colored_mask, 0.5, 0.0, result);

    // Show the result
    cv::imshow("Mask Overlay", result);
    cv::waitKey(0);

    // save the result
    cv::imwrite("F:\\Hacarus\\SAM_zhu\\notebooks\\images\\truck_cppMask.jpg", result);


    return 0;
}
