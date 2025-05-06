import os
from openai import OpenAI
import base64
import time
import json
import requests
from PIL import Image
import io

client = OpenAI(
    api_key= "**",
    base_url="https://api.sensenova.cn/compatible-mode/v1/"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def tensor_to_base64(tensor):
    tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]  
    image = tensor.permute(1, 2, 0).byte().numpy()  
    pil_image = Image.fromarray(image)  

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG") 
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_caption(image):
    if isinstance(image, str):
        base64_image = encode_image(image)
    else:
        base64_image = tensor_to_base64(image)

    completion = client.chat.completions.create(
        model="SenseChat-Vision",
        messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user","content": [
                {"type": "text","text": "Please describe this picture in detail with 70 words. Do not provide any description about feelings."},
                {"type": "image_base64",
                 "image_base64": base64_image}
                ]}],
        max_tokens=77
        )

    # json_data = json.loads(completion.model_dump_json())
    return completion.choices[0].message.content

def get_residual_caption(original_caption, compressed_caption):
    combined_prompt = f"""
    Original Image: '{original_caption}';
    Compressed Image: '{compressed_caption}'.
    Provide information that is in the original image but not included in or mismatch with the compressed image. Don't include information that is already in the compressed image. Please use most compact words. Do not include the description for the compressed image. If you think that the two descriptions mean almost the same thing, please output an empty string. For example: if input is \nOriginal Image: A red barn surrounded by trees, reflected in a pond \nCompressed Image: red house surrounded by trees \nresidual caption is : A barn reflected in a pond. Please refer to this to output. Do not appear words like 'compressed image', 'original image' and 'The semantic residual is'
    """

    completion = client.chat.completions.create(
        model="SenseChat-5",
        messages=[{"role": "system",  "content": "You are a helpful AI assistant. I now need you to extract the semantic residual of two pictures"},
                    {"role": "user",
                   "content": combined_prompt,
                }],
        max_tokens=77
        )

    # json_data = json.loads(completion.model_dump_json())
    return completion.choices[0].message.content