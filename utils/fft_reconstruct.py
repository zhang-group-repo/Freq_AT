import torch
import torch.fft as fft
import math
import pywt
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from io import BytesIO

def fft_reconstruct(images, num_h=0, num_l=10):
    batch_size = len(images)
    shape = images.size()
    w = shape[2]
    h = shape[3]
    lpf = torch.zeros((batch_size, 3, w, h), dtype=torch.float, device='cuda')
    hpf = torch.zeros((batch_size, 3, w, h), dtype=torch.float, device='cuda')
    R_1 = num_l
    R_2 = math.ceil(math.sqrt(w**2 + h**2)) - num_h
    for x in range(w):
        for y in range(h):
            if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R_1**2):
                lpf[:,:,y,x] = 1
    h_pf = 1 - lpf 
    freq = fft.fftn(images,dim=(2,3))
    freq = torch.roll(freq,(h//2,w//2),dims=(2,3)) 
    lpf = freq * lpf
    hpf = freq * h_pf*(torch.rand_like(hpf) * 2 * 0.5 + 1 - 0.5)
    img_l = torch.abs(fft.ifftn(lpf,dim=(2,3)))
    img_h = torch.abs(fft.ifftn(hpf,dim=(2,3)))
#     tensor_img = img_h + img_l
    tensor_img = img_l
    return tensor_img

def jpeg_compress(tensor, quality=70):
    assert len(tensor.shape) == 4, 'tensor must have 4-dimensions (batch, channels, height, width)'

    # Convert tensor to PIL Images
    to_pil = ToPILImage()

    # Compress & Decompress JPEG for every image in the batch
    transformed_batch = []
    for image_tensor in tensor:
        image = to_pil(image_tensor)

        # Compress to JPEG
        with BytesIO() as f:
            image.save(f, format='JPEG', quality=quality)
            f.seek(0)
            
            # Decompress from JPEG
            jpeg_image = Image.open(f)
            tensor_jpeg = ToTensor()(jpeg_image)
            transformed_batch.append(tensor_jpeg)

    return torch.stack(transformed_batch)

def wavelet_transform_multi_level(input_tensor, wavelet='db3', level=1):
    batch_size, channels, height, width = input_tensor.size()
    output = torch.zeros_like(input_tensor.cpu())  # 确保输出张量在CPU上

    for b in range(batch_size):
        for c in range(channels):
            # 将张量转换为NumPy数组
            image = input_tensor[b, c].cpu().numpy()

            # 执行多级离散小波分解
            coeffs = pywt.wavedec2(image, wavelet, level=level)
            cA = coeffs[0]

            # 为忽略的细节系数创建空占位符
            coeffs_modified = [cA] + [(np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)) for cH, cV, cD in coeffs[1:]]

            # 使用修改后的系数列表进行逆变换
            reconstructed = pywt.waverec2(coeffs_modified, wavelet)

            # 适当地裁剪或填充重构的图像以匹配原始尺寸
            output[b, c, :reconstructed.shape[0], :reconstructed.shape[1]] = torch.tensor(reconstructed[:height, :width])

    return output.to(input_tensor.device)  # 确保在返回之前将输出张量移回原始设备
