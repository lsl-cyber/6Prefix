import pandas as pd
import numpy as np
import re
from collections import defaultdict

# -------------------------- 1. 加载数据 --------------------------
df = pd.read_csv("routing_prefix_ipv6_grouped_original_order.csv")
df = df.dropna(subset=["routing_prefix", "IPv6"]).reset_index(drop=True)

# -------------------------- 2. 解析路由前缀和IPv6（2位拆分逻辑） --------------------------
def parse_routing_prefix(prefix):
    """解析路由前缀为段列表+掩码，补全8段4位后拆分为2位子段
    如2001:1203:1000::/36 → 8段4位：['2001','1203','1000','0000','0000','0000','0000','0000'], 36
    → 拆分2位子段：['20','01','12','03','10','00','00','00','00','00','00','00','00','00','00','00']
    """
    prefix_part, mask = prefix.split("/")
    mask = int(mask)
    # 拆分前缀段，去除空段（::表示的省略）
    segments = re.split(":", prefix_part)
    segments = [s for s in segments if s]
    # 补0至8段（IPv6固定8段），每段补前导0至4位
    segments = [s.zfill(4) for s in segments]
    while len(segments) < 8:
        segments.append("0000")
    # 4位拆分为2位子段，生成16段2位网络位
    two_bit_segs = []
    for seg in segments:
        two_bit_segs.append(seg[0:2])
        two_bit_segs.append(seg[2:4])
    return two_bit_segs, mask

def parse_ipv6(ip):
    """解析IPv6为8段4位，再拆分为16段2位十六进制子段，补全前导0"""
    segments = re.split(":", ip)
    segments = [s for s in segments if s]
    # 补0至8段，每段补前导0至4位
    segments = [s.zfill(4) for s in segments]
    while len(segments) < 8:
        segments.append("0000")
    # 4位拆分为2位子段，生成16段2位IP子段
    two_bit_segs = []
    for seg in segments:
        two_bit_segs.append(seg[0:2])
        two_bit_segs.append(seg[2:4])
    return two_bit_segs

# 应用解析：得到16段2位的网络位和16段2位的IP段
df["prefix_segments_2bit"] = df["routing_prefix"].apply(lambda x: parse_routing_prefix(x)[0])
df["prefix_mask"] = df["routing_prefix"].apply(lambda x: parse_routing_prefix(x)[1])
df["ipv6_segments_2bit"] = df["IPv6"].apply(parse_ipv6)

# -------------------------- 3. 构建词表（令牌化：基于2位十六进制子段） --------------------------
# 收集所有唯一令牌（2位网络位子段、2位IP子段、掩码字符串）
all_tokens = set()
for segments in df["prefix_segments_2bit"]:
    all_tokens.update(segments)
for segments in df["ipv6_segments_2bit"]:
    all_tokens.update(segments)
# 掩码整数转换为字符串，避免类型混合
mask_strs = [str(mask) for mask in df["prefix_mask"].unique()]
all_tokens.update(mask_strs)

# 词表：token → id（预留0为PAD，1为UNK）
token2id = defaultdict(lambda: 1)  # 未见过的token映射为UNK（1）
token2id["<PAD>"] = 0
token2id["<UNK>"] = 1
for idx, token in enumerate(sorted(all_tokens), start=2):
    token2id[idx] = token  
token2id = {token: idx for idx, token in enumerate(sorted(all_tokens), start=2)}
token2id["<PAD>"] = 0
token2id["<UNK>"] = 1
id2token = {v: k for k, v in token2id.items()}
vocab_size = len(token2id)

# -------------------------- 4. 编码为模型输入序列 --------------------------
def encode_sequence(segments_2bit, mask):
    """将（16段2位网络位 + 掩码字符串）编码为输入序列：长度17"""
    encoded_segments = [token2id.get(s, token2id["<UNK>"]) for s in segments_2bit]
    encoded_mask = token2id.get(str(mask), token2id["<UNK>"])  # 掩码转为字符串再编码
    return encoded_segments + [encoded_mask]

def encode_ipv6(segments_2bit):
    """将16段2位IP子段编码为目标序列：长度16"""
    return [token2id.get(s, token2id["<UNK>"]) for s in segments_2bit]

# 生成模型输入（src：17段）和目标（tgt：16段）
df["src"] = df.apply(lambda x: encode_sequence(x["prefix_segments_2bit"], x["prefix_mask"]), axis=1)
df["tgt"] = df["ipv6_segments_2bit"].apply(encode_ipv6)

# -------------------------- 5. 保存预处理结果和词表 --------------------------
# 保存编码后的数据集（确保为二维数组）
np.save("src_sequences.npy", np.array(df["src"].tolist(), dtype=object), allow_pickle=True)
np.save("tgt_sequences.npy", np.array(df["tgt"].tolist(), dtype=object), allow_pickle=True)

# 保存词表
import json
with open("token2id.json", "w") as f:
    json.dump(dict(token2id), f)  # defaultdict转为普通dict
with open("id2token.json", "w") as f:
    id2token_str_key = {str(k): v for k, v in id2token.items()}  # id转字符串方便json保存
    json.dump(id2token_str_key, f)

# 打印关键信息
print(f"词表大小：{vocab_size}")
print(f"训练样本数：{len(df)}")
print(f"输入序列长度（16段2位网络位+掩码）：17")
print(f"目标序列长度（16段2位IP）：16")