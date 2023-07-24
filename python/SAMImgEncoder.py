
import json
import logging
import numpy as np
import torch
import onnxruntime
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Tuple
from copy import deepcopy
import cv2


class SAMImgEncoder:
    def __init__(self, config_path) -> None:
        with open(config_path) as f:
            config = json.load(f)
        
        # ResizeLongestSide: Proportional scaling the image.
        self.img_size = config['img_size']

        # set device
        self.device = config['device']
        if config['device'] == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logging.info(f"Using device: {self.device}")

        # Preprocess: Normalize the image.
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(self.device)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(self.device)

        # Quantization
        self.qt = config['qt']
    
    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.img_size)
        # print('apply_image.target_size:', target_size)
        # print("before_pil", image.flatten()[:100])
        # print('to_pil_image(image)', np.array(to_pil_image(image)).flatten()[:100], 
        #       np.array(to_pil_image(image)).shape) # (1200, 1800, 3)
        # print('resize(to_pil_image(image), target_size)', np.array(resize(to_pil_image(image), target_size)).flatten()[:100],
        #         np.array(resize(to_pil_image(image), target_size)).shape) # (683, 1024, 3) hwc

        return np.array(resize(to_pil_image(image), target_size))
    
    def apply_img_cv2(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[1], image.shape[0], self.img_size) # w,h
        # print('apply_image_cv2.target_size:', target_size)
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR) # (1024, 683, 3) whc
        
        # print('apply_image_cv2.resized:', resized.flatten()[:100], resized.shape)
        return resized
    
    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.img_size
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w) # scale the coords from original image size to preprocessed image size.
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    def encode_img(self, image: np.ndarray) -> np.ndarray:

        # resize the img, and convert to tensor
        input_image = self.apply_image(image) # note: image is RGB, HxWxC
        cv2_img = self.apply_img_cv2(image)
        print('PIL: ',input_image.flatten()[:100])
        print('cv2: ',cv2_img.flatten()[:100])
        # plt.imshow(input_image)
        # plt.show()
        # plt.imshow(cv2_img)
        # plt.show()
        # print(input_image.shape, cv2_img.shape)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] # HWC to CHW
        print('transformed_image:', transformed_image.flatten()[:100])
        assert (len(transformed_image.shape) == 4
                and transformed_image.shape[1] == 3
                and max(*transformed_image.shape[2:]) == self.img_size), "Invalid image shape before encode"

        # preprocess: normalize the image
        original_image_size = image.shape[:2]
        input_image = self.preprocess(transformed_image)
        print('self.preprocess:', input_image.shape)
        logging.info(f"Img size for Encoder: {original_image_size}")

        # select encoder or qt_encoder(int8)
        img_encoder_path = "../savedmodel/ImageEncoderViT.onnx"
        img_encoder_path_qt = "../savedmodel/ImageEncoderViT_qt.onnx"

        if self.device == 'cuda':
            providers=['CUDAExecutionProvider']
            logging.info("Using CUDA for ImageEncoder")
        else:
            providers=['CPUExecutionProvider']
            logging.info("Using CPU for ImageEncoder")

        if self.qt:
            self.img_encoderONNX = onnxruntime.InferenceSession(img_encoder_path_qt, providers=providers)
        else:
            self.img_encoderONNX = onnxruntime.InferenceSession(img_encoder_path, providers=providers)

        # encode the image
        input_feed = {'input': input_image.cpu().numpy()}
        features = self.img_encoderONNX.run(None, input_feed)
        print('features[0]:', features[0].shape)
        # features is a dict, so features[0] is the value.

        return features[0]

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        print("encode_img.prepocess.norm:", x.flatten()[:100])

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        print("encode_img.prepocess.pad:", x[0])
        return x
