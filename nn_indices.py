import struct

import torch


# 算术编码常量
NUM_SYMBOLS = 256                       # 字节符号表大小：0~255
_ARITH_STATE_BITS = 32                  # 算术编码器状态位宽
_ARITH_FULL = 1 << _ARITH_STATE_BITS    # 2^32
_ARITH_HALF = _ARITH_FULL >> 1          # 2^31
_ARITH_QUARTER = _ARITH_HALF >> 1       # 2^30
_ARITH_THREE_QUARTER = _ARITH_QUARTER * 3
_ARITH_MAX_TOTAL = 1 << 15              # 计数总和上限，超过后做确定性减半缩放

_HEADER_FMT = '>I'                      # 4 字节大端无符号整数（记录 token 个数）
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)



def _to_id_list(indices_list):
    if indices_list is None:
        return []
    if hasattr(indices_list, 'flatten') and hasattr(indices_list, 'tolist'):
        flat = indices_list.flatten()
        if isinstance(flat, torch.Tensor):
            flat = flat.cpu()
        return [int(x) for x in flat.tolist()]
    if isinstance(indices_list, (list, tuple)):
        out = []
        for x in indices_list:
            out.extend(_to_id_list(x))
        return out
    return [int(indices_list)]



#计数数组转成累积计数数组
def _cumulative(freqs):
    cum = [0]
    s = 0
    for f in freqs:
        s += int(f)
        cum.append(s)
    return cum

#编解码一个符号后更新计数
def _update(freqs, sym):
    freqs[sym] += 1
    if sum(freqs) >= _ARITH_MAX_TOTAL:
        for i, f in enumerate(freqs):
            freqs[i] = max(1, (f + 1) // 2)



#比特打包成字符串
def _pack_bits_to_bytes(bits):
    if not bits:
        return b""
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)

#按位读取字节串
class _BitReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.pos = 0

    def read(self):
        if self.pos >= len(self.payload) * 8:
            self.pos += 1
            return 0
        byte = self.payload[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit



#自适应算术编码/解码
def _arithmetic_encode(symbols):
    freqs = [1] * NUM_SYMBOLS            # 均匀初始先验
    low = 0
    high = _ARITH_FULL - 1
    pending = 0
    out_bits = []

    def emit(bit):
        nonlocal pending
        out_bits.append(bit)
        if pending:
            out_bits.extend([1 - bit] * pending)
            pending = 0

    for sym in symbols:
        cum = _cumulative(freqs)
        total = cum[-1]
        width = high - low + 1
        high = low + (width * cum[sym + 1] // total) - 1
        low = low + (width * cum[sym] // total)

        #归一化
        while True:
            if high < _ARITH_HALF:                       #区间落在下半区->确定输出0
                emit(0)
            elif low >= _ARITH_HALF:                     #区间落在上半区->确定输出1
                emit(1)
                low -= _ARITH_HALF
                high -= _ARITH_HALF
            elif low >= _ARITH_QUARTER and high < _ARITH_THREE_QUARTER:
                pending += 1                             #E3
                low -= _ARITH_QUARTER
                high -= _ARITH_QUARTER
            else:
                break
            low = low * 2
            high = high * 2 + 1

        _update(freqs, sym)               #每个符号都参与计数更新

    pending += 1
    if low < _ARITH_QUARTER:
        emit(0)
    else:
        emit(1)

    return _pack_bits_to_bytes(out_bits), len(out_bits)


def _arithmetic_decode(payload, num_symbols):
    freqs = [1] * NUM_SYMBOLS
    low = 0
    high = _ARITH_FULL - 1
    reader = _BitReader(payload)
    code = 0
    #读入32位作为当前码值
    for _ in range(_ARITH_STATE_BITS):
        code = (code << 1) | reader.read()

    decoded = []
    for _ in range(num_symbols):
        cum = _cumulative(freqs)
        total = cum[-1]
        width = high - low + 1
        #反推码值落在哪个符号区间
        value = ((code - low + 1) * total - 1) // width

        sym = None
        for s in range(NUM_SYMBOLS):
            if cum[s] <= value < cum[s + 1]:
                sym = s
                break
        if sym is None:
            raise RuntimeError("自适应算术解码失败：无法定位符号")

        high = low + (width * cum[sym + 1] // total) - 1
        low = low + (width * cum[sym] // total)

        #与编码端完全相同的归一化流程
        while True:
            if high < _ARITH_HALF:
                pass
            elif low >= _ARITH_HALF:
                low -= _ARITH_HALF
                high -= _ARITH_HALF
                code -= _ARITH_HALF
            elif low >= _ARITH_QUARTER and high < _ARITH_THREE_QUARTER:
                low -= _ARITH_QUARTER
                high -= _ARITH_QUARTER
                code -= _ARITH_QUARTER
            else:
                break
            low = low * 2
            high = high * 2 + 1
            code = code * 2 + reader.read()

        decoded.append(sym)
        _update(freqs, sym)

    return decoded


#对外接口，把 CLIP token id 序列编码为字节串
def encode_indices(indices_list):

    ids = _to_id_list(indices_list)
    num_tokens = len(ids)

    #空序列
    if num_tokens == 0:
        return struct.pack(_HEADER_FMT, 0)

    #每个token拆成"高字节+低字节"，共2*num_tokens个符号
    symbols = []
    for tid in ids:
        if tid < 0 or tid > 0xFFFF:
            raise ValueError(f"token id 超出 16 位范围: {tid}")
        symbols.append((tid >> 8) & 0xFF)   # 高字节
        symbols.append(tid & 0xFF)          # 低字节

    payload, _ = _arithmetic_encode(symbols)
    return struct.pack(_HEADER_FMT, num_tokens) + payload

#把encode_indices产生的字节串解码回 CLIP token id 张量
def decode_indices(encoded_bytes):
    if encoded_bytes is None or len(encoded_bytes) < _HEADER_SIZE:
        return torch.empty((1, 0), dtype=torch.int32)

    num_tokens = struct.unpack(_HEADER_FMT, encoded_bytes[:_HEADER_SIZE])[0]
    if num_tokens == 0:
        return torch.empty((1, 0), dtype=torch.int32)

    payload = encoded_bytes[_HEADER_SIZE:]
    symbols = _arithmetic_decode(payload, num_tokens * 2)

    # 每两个字节还原成一个 token id
    ids = []
    for i in range(num_tokens):
        hi = symbols[2 * i]
        lo = symbols[2 * i + 1]
        ids.append((hi << 8) | lo)
    return torch.tensor([ids], dtype=torch.int32)
