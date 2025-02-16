import cv2
import numpy as np
import os


import cv2
import numpy as np

def postprocess_leaf(original_img_path, mask_img_path, output_path):
    # 1. 加载原图和掩膜图像
    orig = cv2.imread(original_img_path)  # 读取原图（BGR格式）
    if orig is None:
        print(f"❌ 无法加载原图: {original_img_path}")
        return
    mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)  # 读取掩膜（灰度图）
    if mask is None:
        print(f"❌ 无法加载掩膜: {mask_img_path}")
        return

    # 2. 确保 mask 和原图的尺寸一致
    if mask.shape != orig.shape[:2]:
        print(f"⚠️ 尺寸不匹配！调整 mask 大小: {mask.shape} → {orig.shape[:2]}")
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 3. 对掩膜进行二值化处理
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 4. 可选：对掩膜进行形态学处理，去除噪点、平滑边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)

    # 5. 创建三通道的掩膜，确保维度匹配
    mask_3c = cv2.merge([clean_mask, clean_mask, clean_mask])

    # 6. 使用 NumPy 进行像素替换：掩膜为黑色的区域变为白色
    result = np.where(mask_3c == 0, 255, orig)

    # 7. 保存处理后的图像
    cv2.imwrite(output_path, result)
    print(f"✅ 处理完成，结果保存至: {output_path}")



# 假设原图在 'test_data/' 文件夹，掩膜在 'results/' 文件夹
input_folder = 'test_data/leaf_data/'
mask_folder = 'test_data/leaf_results/'
output_folder = 'test_data/final_results/'
os.makedirs(output_folder, exist_ok=True)

for subfolder in os.listdir(input_folder):
    if subfolder.startswith("."):  # 跳过隐藏文件夹
        continue
    for filename in os.listdir(input_folder + subfolder):
        orig_path = os.path.join(input_folder + subfolder, filename)
        # 根据原图名称构造对应的掩膜文件名（注意文件名可能有所不同，根据你的实际情况修改）
        mask_path = os.path.join(mask_folder + subfolder, os.path.splitext(filename)[0] + '.png')
        out_path = os.path.join(output_folder + subfolder, filename)
        postprocess_leaf(orig_path, mask_path, out_path)
