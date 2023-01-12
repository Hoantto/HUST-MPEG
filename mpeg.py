from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_dct as dct
import mpeg_matrix as mm
#DCT中的M和N的参数
MN = 8
def transform_invert(img_, transform_train):
    #由data返回PIL图像
    if 'Normalize' in str(transform_train):
        # 分析transforms里的Normalize
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])  # 广播三个维度 乘标准差 加均值

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C

    # 如果有ToTensor，那么之前数值就会被压缩至0-1之间。现在需要反变换回来，也就是乘255
    if 'ToTensor' in str(transform_train):
        img_ = np.array(img_) * 255

    # 先将np的元素转换为uint8数据类型，然后转换为PIL.Image类型
    if img_.shape[2] == 3:  # 若通道数为3 需要转为RGB类型
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:  # 若通道数为1 需要压缩张量的维度至2D
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_

def zigzag(data):
    (row, col) = data.shape
    if(row != col):
        raise("row is not equal to col!")
    zigzag_matrix = torch.zeros((1, row*col),dtype=torch.int64)
    p = 0
    for index in range(2 * row):
        if(index <= row - 1):
            for i in range(index + 1):
                R = i
                C = index - i
                zigzag_matrix[0, p] = data[R, C]
                p += 1
        if(index > row - 1):
            for i in range(2 * row - index - 1):
                C = row - 1 - i
                R = index - C
                zigzag_matrix[0, p] = data[R, C]
                p += 1
    return zigzag_matrix

def mpeg_main_process():
    #读取I帧文件
    prefix, suffix = (input("Plz input the name of I frame: ").split('.'))
    img = Image.open("../ffmpeg-2022-12-11/bin/"+prefix+"."+suffix).convert('RGB')
    img_size = img.size
    #图像边缘填充
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(padding=(int((img_size[0] % MN + 1) / 2), int((img_size[1] % MN + 1) / 2)),
                       padding_mode='edge'
                       ),
    ])
    img_tensor = img_transform(img)
    img_transform = transforms.Compose([transforms.ToTensor()])
    img = transform_invert(img_tensor, img_transform)
    #img.show()
    img_size = img_tensor.shape
    
    #将RGB图转换到YUV颜色空间 
    img_tensor = img_tensor * 255
    YUV_tensor = torch.zeros_like(img_tensor)
    YUV_tensor[0,:,:] = 0.257 * img_tensor[0,:,:] + 0.504 * img_tensor[1,:,:]+ 0.098 * img_tensor[2,:,:] + 16
    YUV_tensor[1,:,:] = -0.148 * img_tensor[0,:,:] - 0.291 * img_tensor[1,:,:] + 0.439 * img_tensor[2,:,:] + 128
    YUV_tensor[2,:,:] = 0.439 * img_tensor[0,:,:] - 0.368 * img_tensor[1,:,:] - 0.071 * img_tensor[2,:,:] + 128
    #print(YUV_tensor,YUV_tensor.shape)
    img = transform_invert(YUV_tensor, img_transform)
    #img.show()
    
    #将YUV通道切分为8x8的图像块
    YUV_tensor = torch.unsqueeze(YUV_tensor, 0)
    YUV_tensor = YUV_tensor.transpose(0, 1)
    #print(YUV_tensor.shape)
    unfold = torch.nn.Unfold(kernel_size=(MN, MN), stride=MN)
    YUV_split_tensor = unfold(YUV_tensor)
    YUV_split_tensor = torch.unsqueeze(YUV_split_tensor,0).reshape(-1, 3, MN, MN)
    #print(YUV_split_tensor.shape)
    #print(YUV_split_tensor[0])
    
    #每个图像块做DCT变换(直接调用库函数即可)
    DCT_split_tensor = dct.dct_3d(YUV_split_tensor)
    #print(DCT_split_tensor.shape)
    #print(DCT_split_tensor[0])
    
    #对获得的DCT系数做量化
    Qt_DCT_split_tensor_Y = DCT_split_tensor[:, 0, :, :] / torch.tensor(mm.Luminance_Quantization_Matrix).unsqueeze(0).reshape(-1,MN, MN)
    Qt_DCT_split_tensor_Y = Qt_DCT_split_tensor_Y.unsqueeze(1)
    Qt_DCT_split_tensor_UV = DCT_split_tensor[:, 1:, :, :] / torch.tensor(mm.Chroma_Quantization_Matrix).unsqueeze(0).reshape(-1, MN, MN)
    Qt_DCT_split_tensor = torch.cat((Qt_DCT_split_tensor_Y,Qt_DCT_split_tensor_UV),1)
    Qt_DCT_split_tensor = Qt_DCT_split_tensor.type(torch.int64)
    #print(Qt_DCT_split_tensor_Y.shape)
    #print(Qt_DCT_split_tensor_UV.shape)
    print(Qt_DCT_split_tensor.shape)
    print(Qt_DCT_split_tensor[0][0])
    
    #将编码后的系数按照z字形编码排列
    z_list = []
    Qt_DCT_split_tensor = Qt_DCT_split_tensor.reshape(-1, MN, MN)
    
    for matrix_grid in Qt_DCT_split_tensor[:, :, :]:
        z_list.append(zigzag(matrix_grid).tolist())
    
    z_tensor  = torch.tensor(z_list).unsqueeze(0).reshape(-1, 3, MN*MN)
    print(z_tensor.shape)
    print(z_tensor[0][0])
     
    
if __name__ == '__main__':
    mpeg_main_process()