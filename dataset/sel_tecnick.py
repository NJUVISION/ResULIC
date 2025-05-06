import os
from PIL import Image

# 定义输入和输出目录
input_dir = '/workspace/test/DiffEIC/dataset/Tecnick'
output_dir = '/workspace/test/DiffEIC/dataset/Tecnick_768'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有图片文件
for filename in os.listdir(input_dir):
    # 获取完整路径
    input_path = os.path.join(input_dir, filename)
    
    # 只处理图像文件（这里假设是常见的图片格式）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 打开图像
        with Image.open(input_path) as img:
            # 获取图像的宽高
            width, height = img.size

            # 按照短边resize到768
            if width < height:
                new_width = 768
                new_height = int(768 * height / width)
            else:
                new_height = 768
                new_width = int(768 * width / height)

            img_resized = img.resize((new_width, new_height))

            # 进行center crop到768x768
            left = (new_width - 768) / 2
            top = (new_height - 768) / 2
            right = (new_width + 768) / 2
            bottom = (new_height + 768) / 2

            img_cropped = img_resized.crop((left, top, right, bottom))

            # 保存处理后的图像
            output_path = os.path.join(output_dir, filename)
            img_cropped.save(output_path)

            print(f"Processed and saved: {output_path}")

print("Processing completed!")
