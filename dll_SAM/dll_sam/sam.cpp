#include <windows.h>
#include "sam.h"

// 1. Standard Library
#include <array>
#include <vector>

// 2. External Libraries
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


cv::Size get_preprocess_shape(int oldh, int oldw, int long_side_length)
{
    static float scale = static_cast<float>(long_side_length) / std::max(oldh, oldw);
    static int newh = static_cast<int>(std::round(oldh * scale));
    static int neww = static_cast<int>(std::round(oldw * scale));
    return cv::Size(neww, newh);
}


cv::Mat apply_image(const cv::Mat& image, const int& image_size = 1024) {
    static cv::Mat resized_image, padded_image, chw_image;

    static cv::Size target_size = get_preprocess_shape(image.rows, image.cols, image_size);

    cv::resize(image, resized_image, target_size, 0, 0, cv::INTER_LINEAR);

    // Pad to square
    int h = resized_image.rows;
    int w = resized_image.cols;
    int pad_h = image_size - h;
    int pad_w = image_size - w;
    assert(pad_h >= 0 && pad_w >= 0);
    cv::Scalar mean(123.675, 116.28, 103.53);
    cv::copyMakeBorder(resized_image, padded_image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, mean);

    cv::dnn::blobFromImage(padded_image, chw_image, 1.0, cv::Size(), mean, false, false, CV_32F);

    cv::Scalar std(58.395, 57.12, 57.375);
    cv::divide(chw_image, std, chw_image);

    return chw_image;
}


std::vector<float> encode_img(const char* img_path, const wchar_t* img_encoder_path) {

    cv::Mat x = cv::imread(img_path);
    if (x.empty()) {
        throw std::runtime_error("Could not read the image");
    }
    x.convertTo(x, CV_32FC3, 1.0f);

    // apply image: 1. resize, 2. normalize, 3. pad to square, 4. HWC to CHW
    cv::Mat x_img = apply_image(x, 1024);

    std::vector<float> x_img_vec;
    x_img_vec.assign((float*)x_img.data, (float*)x_img.data + x_img.total() * x_img.channels());

    constexpr std::int64_t numInputElements = 3 * 1024 * 1024;
    constexpr std::int64_t numOutputElements = 256 * 64 * 64;
    std::vector<float> input(numInputElements);
    std::vector<float> output(numOutputElements);

    const std::array<std::int64_t, 4> inputShape = { 1, 3, 1024, 1024 };
    const std::array<std::int64_t, 4> outputShape = { 1, 256, 64, 64 };

    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    Ort::SessionOptions ort_session_options;
    session = Ort::Session(env, img_encoder_path, Ort::SessionOptions{ nullptr });

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, x_img_vec.data(), x_img_vec.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), outputShape.data(), outputShape.size());

    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };
    inputName.release();
    outputName.release();

    session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

    return output;
}


// C# call encode_img
extern "C" __declspec(dllexport) int encode_img_sharp(const char* img_path, const char* img_encoder_path, float* outputArray, int* outputSize)
{
    const std::size_t size_needed = strlen(img_encoder_path) + 1;
    wchar_t* wide_encoder_path = (wchar_t*)malloc(size_needed * sizeof(wchar_t));

    if (wide_encoder_path != NULL) {
        std::size_t converted_chars = 0;
        mbstowcs_s(&converted_chars, wide_encoder_path, size_needed, img_encoder_path, _TRUNCATE);

        std::vector<float> image_embedding = encode_img(img_path, wide_encoder_path);

        free(wide_encoder_path);

        // padding output array
        *outputSize = image_embedding.size();
        std::copy(image_embedding.begin(), image_embedding.end(), outputArray);
        return 0;

    }
    else {
        *outputSize = 0;
        return -1;
    }
}


std::vector<float> inference(std::vector<float> coords1, std::vector<float> coords2, std::vector<float> output,
    std::vector<float> image_embedding, const wchar_t* onnx_model_path, cv::Mat original_image) {

    // prepare input tensors: image_embedding, onnx_coord, onnx_label, onnx_mask_input, onnx_has_mask_input, orig_im_size
    static const std::array<std::int64_t, 4> image_embedding_shape = { 1, 256, 64, 64 };
    static const std::array<std::int64_t, 3> onnx_coord_shape = { 1, 2, 2 };

    // 1024/img.cols * coordx, 1024/img.cols * coordy
    float coord_x1 = 1024.0f / static_cast<float>(original_image.cols) * coords1[0];
    float coord_y1 = 1024.0f / static_cast<float>(original_image.cols) * coords1[1];
    float coord_x2 = 1024.0f / static_cast<float>(original_image.cols) * coords2[0];
    float coord_y2 = 1024.0f / static_cast<float>(original_image.cols) * coords2[1];
    std::vector<float> onnx_coord = { coord_x1, coord_y1, coord_x2, coord_y2 };

    const std::array<std::int64_t, 2> onnx_label_shape = { 1, 2 };
    std::vector<float> onnx_label = { 1.0f, 1.0f };

    const std::array<std::int64_t, 4> onnx_mask_input_shape = { 1, 1, 256, 256 };
    std::vector<float> onnx_mask_input(1 * 1 * 256 * 256, 0.0f);

    const std::array<std::int64_t, 1> onnx_has_mask_input_shape = { 1 };
    std::vector<float> onnx_has_mask_input(1.0f, 0.0f);

    const std::array<std::int64_t, 1> orig_im_size_shape = { 2 };
    std::vector<float> orig_im_size = { static_cast<float>(original_image.rows), static_cast<float>(original_image.cols) };


    // set session
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session(nullptr);
    session = Ort::Session(env, onnx_model_path, Ort::SessionOptions{ nullptr });

    const std::size_t num_inputs = session.GetInputCount();

    // Create a vector to hold the input tensors
    std::vector<Ort::Value> inputTensors;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, image_embedding.data(), image_embedding.size(), image_embedding_shape.data(), image_embedding_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, onnx_coord.data(), onnx_coord.size(), onnx_coord_shape.data(), onnx_coord_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, onnx_label.data(), onnx_label.size(), onnx_label_shape.data(), onnx_label_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, onnx_mask_input.data(), onnx_mask_input.size(), onnx_mask_input_shape.data(), onnx_mask_input_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, onnx_has_mask_input.data(), onnx_has_mask_input.size(), onnx_has_mask_input_shape.data(), onnx_has_mask_input_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, orig_im_size.data(), orig_im_size.size(), orig_im_size_shape.data(), orig_im_size_shape.size()));

    const std::array<std::int64_t, 4> masksShape = { 1, 1, original_image.rows, original_image.cols };
    const std::array<std::int64_t, 2> scoreShape = { 1, 1 };
    std::vector<float> score(1);
    const std::array<std::int64_t, 4> low_res_logits = { 1, 1, 256, 256 };
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
    constexpr std::size_t inputNamesSize = 6;
    const std::array<const char*, inputNamesSize> inputNames = { inputName0.get(), inputName1.get(), inputName2.get(), inputName3.get(), inputName4.get(), inputName5.get() };
    Ort::AllocatedStringPtr outputName0 = session.GetOutputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName1 = session.GetOutputNameAllocated(1, ort_alloc);
    Ort::AllocatedStringPtr outputName2 = session.GetOutputNameAllocated(2, ort_alloc);
    constexpr std::size_t outputNamesSize = 3;
    const std::array<const char*, outputNamesSize> outputNames = { outputName0.get() , outputName1.get(), outputName2.get() };

    // run
    Ort::RunOptions runOptions;
    session.Run(runOptions, inputNames.data(), inputTensors.data(), 6, outputNames.data(), outputTensors.data(), 3);

    return output;
}

// C# call inference
extern "C" __declspec(dllexport) int inference_sharp( //__declspec(dllexport)
    const float* coords1, int coords1_size,
    const float* coords2, int coords2_size,
    const float* image_embedding, int image_embedding_size,
    const char* onnx_model_path, const char* img_path,
    float* output, int* output_size)
{
    // input array to std::vector
    std::vector<float> coords1_vec(coords1, coords1 + coords1_size);
    std::vector<float> coords2_vec(coords2, coords2 + coords2_size);
    std::vector<float> image_embedding_vec(image_embedding, image_embedding + image_embedding_size);

    // const char* to wchar_t*
    std::size_t size_needed = strlen(onnx_model_path) + 1;
    wchar_t* wide_onnx_model_path = (wchar_t*)malloc(size_needed * sizeof(wchar_t));
    if (wide_onnx_model_path != NULL) {
        std::size_t converted_chars = 0;
        mbstowcs_s(&converted_chars, wide_onnx_model_path, size_needed, onnx_model_path, _TRUNCATE);
        //mbstowcs(wide_onnx_model_path, onnx_model_path, strlen(onnx_model_path) + 1);

        // call inference
        cv::Mat original_image = cv::imread(img_path);
        const int pixels = original_image.rows * original_image.cols;
        std::vector<float> onnx_output(pixels);

        onnx_output = inference(coords1_vec, coords2_vec, onnx_output,
            image_embedding_vec, wide_onnx_model_path, original_image);

        free(wide_onnx_model_path);

        // padding output array
        std::copy(onnx_output.begin(), onnx_output.end(), output);
        return 0;

    }
    else {
		// deal with error
		*output_size = 0;
        return -1;
	}
}
