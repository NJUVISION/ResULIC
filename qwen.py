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
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# original_caption = 'The image shows a riverside scene during the golden hour with a focus on a twin-spired church in the background. In front of the church, there are several historic buildings. The river in the foreground reflects the buildings and the sky. There is a boat with four people rowing on the river, and another boat can be seen in the distance. The sky is partly cloudy, with the sunlight casting a warm glow on the scene.'
# compressed_caption = 'The image is blurred. The central element appears to be a tall structure. Surrounding the structure, there seem to be other buildings and possibly water in the foreground. The colors in the image suggest a warm, possibly sunset or sunrise lighting.'
# combined_prompt = f"""
#     Original Image: '{original_caption}';
#     Compressed Image: '{compressed_caption}'.
#     Provide information that is in the original image but not included in or mismatch with the compressed image. Don't include information that is already in the compressed image. Please use most compact words. Do not include the description for the compressed image. If you think that the two descriptions mean almost the same thing, please output an empty string.
#     """

# completion = client.chat.completions.create(
#         model="qwen-vl-plus",
#         messages=[{"role": "user","content": [
#                 {"type": "text","text": "这是什么，请用中文回答?"},
#                 {"type": "image_url",
#                  "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}}
#                 ]}]
#         )

# json_data = json.loads(completion.model_dump_json())
# print(json_data['choices'][0]['message']['content'])

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
        model="qwen-vl-plus",
        messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user","content": [
                {"type": "text","text": "Please describe this picture in detail with 100 words. Do not provide any description about feelings."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}],
        max_tokens=77
        )

    json_data = json.loads(completion.model_dump_json())
    return json_data['choices'][0]['message']['content']

def get_residual_caption(original_caption, compressed_caption):
    combined_prompt = f"""
    Original Image: '{original_caption}';
    Compressed Image: '{compressed_caption}'.
    Provide information that is in the original image but not included in or mismatch with the compressed image. Don't include information that is already in the compressed image. Please use most compact words. Do not include the description for the compressed image. If you think that the two descriptions mean almost the same thing, please output an empty string. For example: if input is \nOriginal Image: A red barn surrounded by trees, reflected in a pond \nCompressed Image: red house surrounded by trees \nresidual caption is : A barn reflected in a pond. Please refer to this to output. Do not appear words like 'compressed image', 'original image' and 'The semantic residual is'
    """

    completion = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[{"role": "system",  "content": "You are a helpful AI assistant. I now need you to extract the semantic residual of two pictures"},
                    {"role": "user",
                   "content": combined_prompt,
                }],
        max_tokens=77
        )

    json_data = json.loads(completion.model_dump_json())
    return json_data['choices'][0]['message']['content']


# from http import HTTPStatus
# import os
# import dashscope


# def call_with_messages():
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
#                 {"text": "这是什么，请用中文回答?"}
#             ]
#         }
#     ]

#     response = dashscope.MultiModalConversation.call(
#         # export DASHSCOPE_API_KEY ="sk-669fb290d7ef492dbbdb79b0dbbfd733",
#         api_key=os.getenv('sk-669fb290d7ef492dbbdb79b0dbbfd733'),
#         model='llama3.2-90b-vision-instruct',
#         messages=messages,
#     )
#     if response.status_code == HTTPStatus.OK:
#         print(response.output.choices[0].message.content[0]["text"])
#     else:
#         print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
#             response.request_id, response.status_code,
#             response.code, response.message
#         ))


# if __name__ == '__main__':
#     call_with_messages()