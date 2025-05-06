import torch
import torchac
import numpy as np
import prompt_inversion.open_clip as open_clip 

import zlib

def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()
    texts = []
    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))
            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))
    return texts

def calculate_char_frequencies(indices_list):

    text = str(indices_list).replace(' ', '').replace('[', '').replace(']', '')

    char_freq = {}
    for char in text:

        if char.isdigit() or char == ',':
            char_freq[char] = char_freq.get(char, 0) + 1

    # print("字符频率:")
    # for char, freq in sorted(char_freq.items()):
    #     print(f"{repr(char)}: {freq}")
    
    return char_freq

    
#     return cdf, unique_chars
def create_cdf_from_frequencies(char_freq, sequence_length):
    """
    根据字符频率创建累积分布函数，扩展到 sequence_length。
    """
    total_chars = sum(char_freq.values())

    unique_chars = sorted(char_freq.keys())

    prob = [char_freq.get(char, 0) / total_chars for char in unique_chars]
    
    # 累积分布（归一化到 [0, 1] 范围）
    cdf = torch.zeros(len(unique_chars) + 1, dtype=torch.float32)
    cdf[1:] = torch.cumsum(torch.tensor(prob, dtype=torch.float32), dim=0)
    cdf[-1] = 1.0  # 保证最后一个值是 1.0
    
    # 扩展到 sequence_length
    cdf = cdf.view(1, 1, -1).expand(1, sequence_length, -1)
    
    return cdf, unique_chars
def arithmetic_encode(indices_list):
    """
    使用数字和逗号频率进行算术编码
    """
    # 计算字符频率
    char_freq = calculate_char_frequencies(indices_list)
    text = ','.join(map(str, indices_list.flatten().tolist()))
    # 创建CDF
    sequence_length = len(text)
    # print(f"sequence_length: {sequence_length}")
    cdf, unique_chars = create_cdf_from_frequencies(char_freq, sequence_length)
    cdf = cdf.squeeze(0)
    # cdf, unique_chars = create_cdf_from_frequencies(char_freq)
    # print(f"unique_chars: {unique_chars}")
    # try:
    # 将输入转换为字符串，移除空格和方括号
    # text = str(indices_list).replace(' ', '').replace('[', '').replace(']', '')
    

    # print(f"text:{text}")
    # 将字符映射到索引
    sym = torch.tensor([unique_chars.index(char) for char in text], dtype=torch.int16)
    sym = sym.unsqueeze(0)
    # print(f"sym:{sym}")
    # 编码
    encoded_bytes = torchac.encode_float_cdf(cdf.unsqueeze(0), sym, check_input_bounds=True)
    
    return encoded_bytes, cdf, unique_chars

    # except Exception as e:
    #     print(f"编码错误: {e}")
    #     return None, None, None

def arithmetic_decode(encoded_bytes, cdf, unique_chars):
    """
    解码
    """
    try:
        # 解码
        decoded_indices = torchac.decode_float_cdf(cdf, encoded_bytes)
        
        # 将索引转换回字符
        decoded_text = ''.join(unique_chars[idx] for idx in decoded_indices)
        
        return decoded_text
    
    except Exception as e:
        print(f"解码错误: {e}")
        return None


# prompt = 'Someone is holding a phone while watching a video on it'
# tokenizer = open_clip.tokenizer._tokenizer
# prompt_ids = tokenizer.encode(prompt)
# best_ids = torch.tensor([prompt_ids], device='cuda')
# print(best_ids)

# zlib_compress = zlib.compress(prompt.encode('utf-8'), level=zlib.Z_BEST_COMPRESSION)
# print(f'text_bits_zlib: {len(zlib_compress) * 8}')

# encoded_data, cdf, unique_chars = arithmetic_encode(best_ids)
# print(f'encoded_data: {encoded_data}')
# print(f'text_bits: {len(encoded_data) * 8}')

# decoded_text = arithmetic_decode(encoded_data, cdf, unique_chars)
# decoded_text = torch.tensor(list(map(int, decoded_text.split(',')))).unsqueeze(0).to('cuda')

# decoded_text = decode_ids(decoded_text, tokenizer)[0]
# print(f'decoded_text: {decoded_text}')