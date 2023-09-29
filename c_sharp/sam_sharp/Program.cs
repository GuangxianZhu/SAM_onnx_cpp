// See https://aka.ms/new-console-template for more information
using System.Diagnostics.Tracing;
using System.Runtime.InteropServices;
using OpenCvSharp;
using OpenCvSharp.Extensions;

int clickState = 0; // Global state variable
float[] coords1 = new float[2];
float[] coords2 = new float[2];

void CallBackFunc(MouseEventTypes eventTypes, int x, int y, MouseEventFlags flags, IntPtr userdata)
{
    if (eventTypes == MouseEventTypes.LButtonDown)
    {
        if (clickState == 0)
        {
            coords1[0] = x;
            coords1[1] = y;
            Console.WriteLine($"First click - position ({x}, {y})");
        }
        else
        {
            coords2[0] = x;
            coords2[1] = y;
            Console.WriteLine($"Second click - position ({x}, {y})");
        }
        clickState = 1 - clickState; // Toggle the state between 0 and 1
    }
}


void Main() {

    float[] encodeOutputArray = new float[1 * 256 * 64 * 64]; // make sure the size is this.
    int encodeOutputSize;

    // test encode_img_sharp
    int err_flag_encode = SAM.encode_img_sharp(
        "F:\\Hacarus\\SAM_zhu\\notebooks\\images\\groceries.jpg",
        "F:\\Hacarus\\SAM_zhu\\savedmodel\\ImageEncoderViT.onnx",
        encodeOutputArray, out encodeOutputSize);
    if (err_flag_encode == -1) {         
        Console.WriteLine("image encoder path error");
       }
    Console.WriteLine("encode_img_sharp's  size: " + encodeOutputSize);

    // user click to get coords1 and coords2
    string imgPath = "F:\\Hacarus\\SAM_zhu\\notebooks\\images\\groceries.jpg";
    Mat originalImage = Cv2.ImRead(imgPath);
    Window window = new Window("Image");
    window.SetMouseCallback(CallBackFunc);

    // show text on image prompt user to click
    Cv2.PutText(originalImage, "Click twice to select the region, then press A", 
        new Point(10, 30), HersheyFonts.HersheyPlain, 2.0, new Scalar(0, 255, 0), 2);

    while (true)
    {
        Cv2.ImShow("Image", originalImage);
        string onnx_model_path = "F:\\Hacarus\\SAM_zhu\\savedmodel\\sam_onnx_example.onnx";
        string img_path = "F:\\Hacarus\\SAM_zhu\\notebooks\\images\\groceries.jpg";
        //opencv read image, return img size, to create output array.
        Mat img = Cv2.ImRead(img_path);
        int img_size = img.Width * img.Height;
        float[] output = new float[img_size]; // make sure the size is origin image size.
        int output_size;
        // inference
        int err_flag_infer = SAM.inference_sharp(
               coords1, coords1.Length,
                  coords2, coords2.Length,
                     encodeOutputArray, encodeOutputSize,
                        onnx_model_path, img_path,
                           output, out output_size);
        if (err_flag_infer == -1) {
            Console.WriteLine("mask generator path error");
        }

        // cv imshow the output
        Mat output_mask = new Mat(img.Height, img.Width, MatType.CV_32FC1, output);
        // convert to 8UC1
        output_mask.ConvertTo(output_mask, MatType.CV_8UC1);
        // Normalize the mask to range between 0 and 255
        Cv2.Normalize(output_mask, output_mask, 0, 255, NormTypes.MinMax);
        // Resize the mask to the size of original image
        Cv2.Resize(output_mask, output_mask, new Size(img.Width, img.Height));
        // threshold the mask
        Cv2.Threshold(output_mask, output_mask, 0, 255, ThresholdTypes.Binary);
        // Create a colored mask, light blue [255, 229, 204]
        Mat colored_mask = new Mat(img.Height, img.Width, MatType.CV_8UC3);
        colored_mask.SetTo(new Scalar(255, 229, 204), output_mask);
        // overlay the colored mask onto original image
        Mat overlayed_img = new Mat();
        Cv2.AddWeighted(img, 0.5, colored_mask, 0.5, 0, overlayed_img);
        // show the overlayed image
        Cv2.ImShow("overlayed_img", overlayed_img);
        
        // clear the coords
        coords1 = new float[2];
        coords2 = new float[2];

        if (Cv2.WaitKey(0) == (int)'q')
            break;
    }

    Cv2.DestroyAllWindows();

}

Main();

///////////////////////////////////////////////////////////////////////


// do not change function name
class SAM
{
    // F:\\Hacarus\\dll_sam\\x64\\Debug\\dll_sam.dll
    [DllImport("sam_cxx.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int encode_img_sharp(
        string img_path, string img_encoder_path,
        float[] outputArray, out int outputSize);

    [DllImport("sam_cxx.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int inference_sharp(
        float[] coords1, int coords1_size,
        float[] coords2, int coords2_size,
        float[] image_embedding, int image_embedding_size,
        string onnx_model_path, string img_path,
        float[] output, out int output_size);
}

