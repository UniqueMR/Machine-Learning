import numpy as np

# 配置文件
config = {
    # 训练集文件
    'train_images_idx3_ubyte_file_path': 
    'data/train-images.idx3-ubyte',
    # 训练集标签文件
    'train_labels_idx1_ubyte_file_path': 
    'data/train-labels.idx1-ubyte',

    # 测试集文件
    'test_images_idx3_ubyte_file_path': 
    'data/t10k-images.idx3-ubyte',
    # 测试集标签文件
    'test_labels_idx1_ubyte_file_path': 
    'data/t10k-labels.idx1-ubyte',

    # 特征提取阙值
    'binarization_limit_value': 0.14,

    # 特征提取后的边长
    'side_length': 14
}

def oneImagesFeatureExtraction(image):
    '''
    对单张图片进行特征提取
    '''
    res = np.empty((config['side_length'], config['side_length']))
    num = 28//config['side_length']
    for i in range(0, config['side_length']):
        for j in range(0, config['side_length']):
            # tempMean = (image[2*i:2*(i+1),2*j:2*(j+1)] != 0).sum()/(2 * 2)
            tempMean = image[num*i:num*(i+1), num*j:num*(j+1)].mean()
            if tempMean > config['binarization_limit_value']:
                res[i, j] = 1
            else:
                res[i, j] = 0
    return res


def featureExtraction(images):
    """
        对所有图片进行特征提取
    """
    res = np.empty((images.shape[0], config['side_length'],
                    config['side_length']), dtype=np.float32)
    for i in range(images.shape[0]):
        res[i] = oneImagesFeatureExtraction(images[i])
    return res
