import torch
import torch.nn as nn
import numpy as np
import json
import re
import pandas as pd
import ipaddress
import warnings
import socket
import subprocess
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

# -------------------------- 1. 设备与生成配置 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 总正式生成预算
TOTAL_GENERATION_BUDGET = 2000000

# 预探测，用于探测估计命中率
PREPROBE_RATIO = 0.001
PREPROBE_NUM_PER_PREFIX = max(1, int(TOTAL_GENERATION_BUDGET * PREPROBE_RATIO))

# 采样生成参数：保持原脚本生成逻辑，仅调整批量分配方式
TEMPERATURE = 0.7
TOP_K = 30
MAX_ATTEMPTS_MULTIPLIER = 10
MIN_MAX_ATTEMPTS = 1000

# 预算分配平滑项。

# 预探测协议配置：任一协议探测成功即视为命中
ENABLE_PREPROBE_PROBING = True
PROBE_PROTOCOLS = ("icmpv6", "tcp80", "tcp443")
PROBE_TIMEOUT = 1.0
PROBE_WORKERS = 128

# 文件配置
PREFIX_FILE = "unique_routing_prefixes.csv"
MODEL_FILE = "transformer_ip_generator.pth"
TOKEN2ID_FILE = "token2id.json"
ID2TOKEN_FILE = "id2token.json"
OUTPUT_FILE = "generated_ipv6.csv"
PREPROBE_OUTPUT_FILE = "preprobe_generated_ipv6.csv"
PREPROBE_STATS_FILE = "preprobe_stats.csv"
ALLOCATION_FILE = "budget_allocation.csv"


# -------------------------- 2. 定义模型类（与 train.py 保持一致） --------------------------
class TransformerIPGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_pos_encoder(self, seq_len, d_model):
        """动态生成任意长度的位置编码。"""
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-np.log(10000.0) / d_model))
        pos_enc = torch.zeros(1, seq_len, d_model, device=device)
        pos_enc[0, :, 0::2] = torch.sin(pos * div_term)
        pos_enc[0, :, 1::2] = torch.cos(pos * div_term)
        return pos_enc

    def generate_masks(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_key_padding_mask = (src == self.embedding.padding_idx).to(device)
        tgt_key_padding_mask = (tgt == self.embedding.padding_idx).to(device)
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()
        src_mask = None

        return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask

    def forward(self, src, tgt, token2id=None):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_pos_enc = self._generate_pos_encoder(src_seq_len, self.d_model)
        tgt_pos_enc = self._generate_pos_encoder(tgt_seq_len, self.d_model)

        src_emb = self.dropout(self.embedding(src) + src_pos_enc)
        tgt_emb = self.dropout(self.embedding(tgt) + tgt_pos_enc)

        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.generate_masks(src, tgt)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = self.fc_out(out)
        return out


# -------------------------- 3. 基础辅助函数 --------------------------
def parse_routing_prefix(prefix):
    """解析路由前缀为 16 段 2 位十六进制子段列表 + 掩码。"""
    prefix_part, mask = prefix.split("/")
    mask = int(mask)

    segments = re.split(":", prefix_part)
    segments = [s for s in segments if s]
    segments = [s.zfill(4) for s in segments]
    while len(segments) < 8:
        segments.append("0000")

    two_bit_segs = []
    for seg in segments:
        two_bit_segs.append(seg[0:2])
        two_bit_segs.append(seg[2:4])
    return two_bit_segs, mask


def encode_sequence(segments_2bit, mask, token2id):
    """将（16 段 2 位网络位 + 掩码字符串）编码为输入序列：长度 17。"""
    encoded_segments = [token2id.get(s, token2id["<UNK>"]) for s in segments_2bit]
    encoded_mask = token2id.get(str(mask), token2id["<UNK>"])
    return encoded_segments + [encoded_mask]


def preprocess_target_prefix(prefix_str, token2id):
    """将目标路由前缀编码为模型输入序列（长度 17）。"""
    segments_2bit, mask = parse_routing_prefix(prefix_str)
    encoded = encode_sequence(segments_2bit, mask, token2id)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)


def decode_ipv6(encoded_seq, token2id, id2token):
    """将编码后的 2 位子段序列解码为 IPv6 地址字符串。"""
    two_bit_segs = []
    for idx in encoded_seq:
        idx = int(idx)
        if idx == token2id["<PAD>"] or idx == token2id["<UNK>"]:
            continue
        if idx not in id2token:
            continue
        two_bit_segs.append(id2token[idx])
        if len(two_bit_segs) >= 16:
            break

    while len(two_bit_segs) < 16:
        two_bit_segs.append("00")

    four_bit_segs = []
    for i in range(0, 16, 2):
        four_bit = two_bit_segs[i] + two_bit_segs[i + 1]
        four_bit_segs.append(four_bit.lstrip("0") if four_bit.lstrip("0") else "0")

    try:
        ipv6_full = ":".join(four_bit_segs)
        ipv6_str = str(ipaddress.IPv6Address(ipv6_full))
        return ipv6_str
    except Exception:
        return ":".join(four_bit_segs)


# -------------------------- 4. 合法性校验函数 --------------------------
def is_ip_in_prefix(ip_str, prefix_str):
    """验证 IPv6 地址是否属于目标路由前缀。"""
    try:
        ipv6 = ipaddress.IPv6Address(ip_str)
        prefix = ipaddress.IPv6Network(prefix_str, strict=False)
        return ipv6 in prefix
    except Exception:
        return False


# -------------------------- 5. 单地址生成函数 --------------------------
def generate_model_only_ip(prefix_str, token2id, id2token, model, max_len=16, temperature=0.7, top_k=20):
    prefix_segments_2bit, mask = parse_routing_prefix(prefix_str)
    network_bits = mask
    bits_per_two_bit_seg = 8
    fixed_two_bit_count = min((network_bits + bits_per_two_bit_seg - 1) // bits_per_two_bit_seg, 16)

    prefix_tensor = preprocess_target_prefix(prefix_str, token2id)
    tgt_seq = torch.full((1, max_len), token2id["<PAD>"], dtype=torch.long).to(device)

    for i in range(min(fixed_two_bit_count, max_len)):
        if i < len(prefix_segments_2bit):
            seg_2bit = prefix_segments_2bit[i]
            tgt_seq[:, i] = token2id.get(seg_2bit, token2id["<UNK>"])
        else:
            tgt_seq[:, i] = token2id.get("00", token2id["<PAD>"])

    # 若前缀极短导致没有固定字段，则给解码器一个起始字段，避免空目标序列输入
    if fixed_two_bit_count == 0:
        tgt_seq[:, 0] = token2id.get("00", token2id["<PAD>"])
        start_idx = 1
    else:
        start_idx = min(fixed_two_bit_count, max_len - 1)

    with torch.no_grad():
        for i in range(start_idx, max_len):
            output = model(prefix_tensor, tgt_seq[:, :i])
            logits = output[:, -1, :] / temperature

            if top_k > 0:
                current_top_k = min(top_k, logits.shape[-1])
                top_k_values, top_k_indices = torch.topk(logits, current_top_k)
                probs = torch.nn.functional.softmax(top_k_values, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(1, next_token_idx).squeeze(1)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            tgt_seq[:, i] = next_token

    encoded_ip = tgt_seq.squeeze(0).cpu().numpy()
    ip_str = decode_ipv6(encoded_ip, token2id, id2token)
    return ip_str


# -------------------------- 6. 多地址生成函数（保留原逻辑，增加 exclude_ips 去重约束） --------------------------
def generate_multi_ips_for_prefix(prefix_str, token2id, id2token, model, max_num=1000,
                                  max_attempts=10000, temperature=0.7, top_k=20,
                                  exclude_ips=None, log_every=50):
    valid_ips = []
    valid_ip_set = set()
    exclude_ips = exclude_ips if exclude_ips is not None else set()
    attempt_count = 0
    dynamic_temp = temperature

    while len(valid_ips) < max_num and attempt_count < max_attempts:
        attempt_count += 1

        if attempt_count % 100 == 0:
            dynamic_temp = max(0.5, dynamic_temp * 0.95)

        try:
            ip_str = generate_model_only_ip(
                prefix_str, token2id, id2token, model,
                max_len=16, temperature=dynamic_temp, top_k=top_k
            )
        except IndexError as e:
            if attempt_count % 500 == 0:
                print(f"  生成 IP 时触发索引越界：{e}（前缀 {prefix_str}，尝试次数 {attempt_count}）")
            continue
        except Exception as e:
            if attempt_count % 500 == 0:
                print(f"  生成 IP 失败：{e}（前缀 {prefix_str}，尝试次数 {attempt_count}）")
            continue

        if is_ip_in_prefix(ip_str, prefix_str) and ip_str not in valid_ip_set and ip_str not in exclude_ips:
            valid_ips.append(ip_str)
            valid_ip_set.add(ip_str)
            if log_every and len(valid_ips) % log_every == 0:
                print(f"  已生成 {len(valid_ips)}/{max_num} 个合法 IP（尝试 {attempt_count} 次，当前温度 {dynamic_temp:.2f}）")

    if len(valid_ips) < max_num:
        print(f"  警告：仅生成 {len(valid_ips)} 个 IP（目标 {max_num}，尝试 {attempt_count}/{max_attempts} 次）")
    else:
        print(f"  完成生成：共尝试 {attempt_count} 次，成功生成 {len(valid_ips)}/{max_num} 个合法 IP")
    return valid_ips


# -------------------------- 7. 预探测函数 --------------------------
def probe_icmpv6(ip_str, timeout=1.0):
    """通过系统 ping 命令进行 ICMPv6 探测。"""
    ping_cmd = shutil.which("ping") or shutil.which("ping6")
    if ping_cmd is None:
        return False

    system_name = platform.system().lower()
    try:
        if "windows" in system_name:
            cmd = [ping_cmd, "-6", "-n", "1", "-w", str(int(timeout * 1000)), ip_str]
        elif "darwin" in system_name:
            cmd = [ping_cmd, "-6", "-c", "1", "-W", str(int(timeout * 1000)), ip_str]
        else:
            cmd = [ping_cmd, "-6", "-c", "1", "-W", str(max(1, int(timeout))), ip_str]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout + 2)
        return result.returncode == 0
    except Exception:
        return False


def probe_tcp_port(ip_str, port, timeout=1.0):
    """通过 TCP connect 进行端口探测。"""
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            ret = sock.connect_ex((ip_str, port, 0, 0))
            return ret == 0
    except Exception:
        return False


def probe_active_ipv6(ip_str, timeout=1.0, protocols=PROBE_PROTOCOLS):
    """多协议活跃性探测：ICMPv6、TCP/80、TCP/443 任一成功即视为活跃。"""
    if "icmpv6" in protocols and probe_icmpv6(ip_str, timeout=timeout):
        return True
    if "tcp80" in protocols and probe_tcp_port(ip_str, 80, timeout=timeout):
        return True
    if "tcp443" in protocols and probe_tcp_port(ip_str, 443, timeout=timeout):
        return True
    return False


def probe_ips_parallel(ip_list, timeout=1.0, workers=128):
    """并行探测一组 IPv6 地址，返回活跃地址集合。"""
    active_ips = set()
    if not ip_list:
        return active_ips

    workers = max(1, min(workers, len(ip_list)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ip = {
            executor.submit(probe_active_ipv6, ip, timeout, PROBE_PROTOCOLS): ip
            for ip in ip_list
        }
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                if future.result():
                    active_ips.add(ip)
            except Exception:
                pass
    return active_ips


# -------------------------- 8. 词表与模型加载 --------------------------
def load_vocab_and_model():
    """加载词表和训练好的模型。"""
    try:
        with open(TOKEN2ID_FILE, "r") as f:
            token2id = json.load(f)
        with open(ID2TOKEN_FILE, "r") as f:
            id2token = json.load(f)
            id2token = {int(k): v for k, v in id2token.items()}
    except FileNotFoundError as e:
        print(f"词表文件缺失：{e}")
        return None, None, None

    vocab_size = len(token2id)
    model = TransformerIPGenerator(vocab_size, d_model=128, nhead=8, num_layers=6, dropout=0.1).to(device)
    model.embedding.padding_idx = token2id["<PAD>"]

    try:
        try:
            state_dict = torch.load(MODEL_FILE, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(MODEL_FILE, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError as e:
        print(f"模型权重文件缺失：{e}")
        return None, None, None

    return token2id, id2token, model


# -------------------------- 9. 预算分配函数 --------------------------
def allocate_budget_by_hit_rate(prefixes, preprobe_stats, total_budget):
    """根据每条前缀的预探测命中率按比例分配正式生成预算。"""
    if not prefixes:
        return {}

    rates = []
    for prefix in prefixes:
        hit_rate = float(preprobe_stats.get(prefix, {}).get("hit_rate", 0.0))
        rates.append(max(0.0, hit_rate) + BUDGET_SMOOTH_EPS)

    rates = np.array(rates, dtype=np.float64)

    # 若全部前缀预探测命中率为 0，则退化为均匀分配，避免无法生成正式地址
    if rates.sum() <= 0:
        weights = np.ones(len(prefixes), dtype=np.float64) / len(prefixes)
        allocation_reason = "uniform_fallback_all_zero_hit_rate"
    else:
        weights = rates / rates.sum()
        allocation_reason = "proportional_to_preprobe_hit_rate"

    raw_alloc = weights * total_budget
    base_alloc = np.floor(raw_alloc).astype(int)
    remainder = int(total_budget - base_alloc.sum())

    if remainder > 0:
        fractional = raw_alloc - base_alloc
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            base_alloc[idx] += 1

    allocation = {}
    for prefix, budget, weight, raw in zip(prefixes, base_alloc, weights, raw_alloc):
        allocation[prefix] = {
            "allocated_budget": int(budget),
            "allocation_weight": float(weight),
            "raw_allocated_budget": float(raw),
            "allocation_reason": allocation_reason
        }
    return allocation


# -------------------------- 10. 批量生成主逻辑 --------------------------
def main():
    token2id, id2token, model = load_vocab_and_model()
    if token2id is None or id2token is None or model is None:
        return

    try:
        with open(PREFIX_FILE, "r") as f:
            target_prefixes = [line.strip() for line in f if line.strip()]
        # 保持顺序去重
        target_prefixes = list(dict.fromkeys(target_prefixes))
        if not target_prefixes:
            print("目标路由前缀文件为空！")
            return
    except FileNotFoundError:
        print("错误：未找到前缀路由文件，请先创建该文件！")
        print("示例内容（每行一个前缀）：")
        print("2001:4860:4860::/32")
        print("2400:3200::/12")
        return

    print(f"\n开始目标生成任务：目标前缀数 {len(target_prefixes)}，正式生成总预算 {TOTAL_GENERATION_BUDGET}")
    print(f"预探测配置：每条前缀先生成 {PREPROBE_NUM_PER_PREFIX} 个地址进行探测")
    print(f"生成配置：初始温度 {TEMPERATURE}，Top-K={TOP_K}")

    # -------------------------- 10.1 预生成 + 预探测 --------------------------
    preprobe_records = []
    preprobe_stats = {}
    preprobe_seen_ips = set()

    print("\n========== 阶段一：预探测 ==========")
    for idx, prefix in enumerate(target_prefixes, 1):
        print(f"\n[预探测 {idx}/{len(target_prefixes)}] 前缀：{prefix}")
        pre_max_attempts = max(MIN_MAX_ATTEMPTS, PREPROBE_NUM_PER_PREFIX * MAX_ATTEMPTS_MULTIPLIER)
        pre_ips = generate_multi_ips_for_prefix(
            prefix, token2id, id2token, model,
            max_num=PREPROBE_NUM_PER_PREFIX,
            max_attempts=pre_max_attempts,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            exclude_ips=preprobe_seen_ips,
            log_every=0
        )
        preprobe_seen_ips.update(pre_ips)

        if ENABLE_PREPROBE_PROBING and pre_ips:
            print(f"  开始探测 {len(pre_ips)} 个预生成地址 ...")
            active_ips = probe_ips_parallel(pre_ips, timeout=PROBE_TIMEOUT, workers=PROBE_WORKERS)
        else:
            active_ips = set()

        hit_count = len(active_ips)
        generated_count = len(pre_ips)
        hit_rate = hit_count / generated_count if generated_count > 0 else 0.0

        print(f"  预探测完成：生成 {generated_count}，命中 {hit_count}，命中率 {hit_rate:.6f}")

        preprobe_stats[prefix] = {
            "preprobe_generated": generated_count,
            "preprobe_hits": hit_count,
            "hit_rate": hit_rate
        }

        for ip in pre_ips:
            preprobe_records.append({
                "routing_prefix": prefix,
                "generated_ipv6": ip,
                "is_active": 1 if ip in active_ips else 0,
                "generation_source": "preprobe_model_sampling"
            })

    if preprobe_records:
        pd.DataFrame(preprobe_records).to_csv(PREPROBE_OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\n预探测地址明细已保存到：{PREPROBE_OUTPUT_FILE}")

    preprobe_stats_df = pd.DataFrame([
        {
            "routing_prefix": prefix,
            "preprobe_generated": stat["preprobe_generated"],
            "preprobe_hits": stat["preprobe_hits"],
            "hit_rate": stat["hit_rate"]
        }
        for prefix, stat in preprobe_stats.items()
    ])
    preprobe_stats_df.to_csv(PREPROBE_STATS_FILE, index=False, encoding="utf-8")
    print(f"预探测统计结果已保存到：{PREPROBE_STATS_FILE}")

    # -------------------------- 10.2 根据预探测命中率分配正式生成预算 --------------------------
    print("\n========== 阶段二：按预探测命中率分配正式生成预算 ==========")
    allocation = allocate_budget_by_hit_rate(target_prefixes, preprobe_stats, TOTAL_GENERATION_BUDGET)

    allocation_records = []
    for prefix in target_prefixes:
        stat = preprobe_stats.get(prefix, {})
        alloc = allocation.get(prefix, {})
        allocation_records.append({
            "routing_prefix": prefix,
            "preprobe_generated": stat.get("preprobe_generated", 0),
            "preprobe_hits": stat.get("preprobe_hits", 0),
            "hit_rate": stat.get("hit_rate", 0.0),
            "allocation_weight": alloc.get("allocation_weight", 0.0),
            "allocated_budget": alloc.get("allocated_budget", 0),
            "allocation_reason": alloc.get("allocation_reason", "")
        })

    allocation_df = pd.DataFrame(allocation_records)
    allocation_df.to_csv(ALLOCATION_FILE, index=False, encoding="utf-8")
    print(f"预算分配结果已保存到：{ALLOCATION_FILE}")
    print(f"正式预算分配总和：{allocation_df['allocated_budget'].sum()}")

    # -------------------------- 10.3 按分配预算正式生成地址 --------------------------
    print("\n========== 阶段三：正式目标生成 ==========")
    generated_records = []
    generated_seen_ips = set(preprobe_seen_ips)  

    for idx, prefix in enumerate(target_prefixes, 1):
        budget = int(allocation[prefix]["allocated_budget"])
        if budget <= 0:
            print(f"\n[正式生成 {idx}/{len(target_prefixes)}] 前缀：{prefix}，分配预算为 0，跳过")
            continue

        print(f"\n[正式生成 {idx}/{len(target_prefixes)}] 前缀：{prefix}，分配预算：{budget}")
        formal_max_attempts = max(MIN_MAX_ATTEMPTS, budget * MAX_ATTEMPTS_MULTIPLIER)
        valid_ip_list = generate_multi_ips_for_prefix(
            prefix, token2id, id2token, model,
            max_num=budget,
            max_attempts=formal_max_attempts,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            exclude_ips=generated_seen_ips,
            log_every=50
        )

        for ipv6 in valid_ip_list:
            generated_seen_ips.add(ipv6)
            generated_records.append({
                "routing_prefix": prefix,
                "generated_ipv6": ipv6,
                "generation_source": "model_sampling_weighted_budget"
            })

    remaining = TOTAL_GENERATION_BUDGET - len(generated_records)
    if remaining > 0:
        print(f"\n正式生成数量不足目标预算，还差 {remaining} 个地址，开始按命中率从高到低补足 ...")
        sorted_prefixes = sorted(
            target_prefixes,
            key=lambda p: preprobe_stats.get(p, {}).get("hit_rate", 0.0),
            reverse=True
        )
        no_progress_rounds = 0
        while remaining > 0 and no_progress_rounds < 2:
            progress = 0
            for prefix in sorted_prefixes:
                if remaining <= 0:
                    break
                request_num = min(remaining, max(1, int(allocation[prefix]["allocated_budget"])))
                if request_num <= 0:
                    request_num = min(remaining, PREPROBE_NUM_PER_PREFIX)
                print(f"\n[补足生成] 前缀：{prefix}，目标补足：{request_num}，剩余缺口：{remaining}")
                valid_ip_list = generate_multi_ips_for_prefix(
                    prefix, token2id, id2token, model,
                    max_num=request_num,
                    max_attempts=max(MIN_MAX_ATTEMPTS, request_num * MAX_ATTEMPTS_MULTIPLIER),
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    exclude_ips=generated_seen_ips,
                    log_every=50
                )
                for ipv6 in valid_ip_list:
                    generated_seen_ips.add(ipv6)
                    generated_records.append({
                        "routing_prefix": prefix,
                        "generated_ipv6": ipv6,
                        "generation_source": "model_sampling_weighted_budget_topup"
                    })
                progress += len(valid_ip_list)
                remaining = TOTAL_GENERATION_BUDGET - len(generated_records)
            if progress == 0:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0

    # -------------------------- 10.4 保存正式生成结果 --------------------------
    if generated_records:
        generated_df = pd.DataFrame(generated_records)
        generated_df = generated_df.drop_duplicates(subset=["generated_ipv6"]).reset_index(drop=True)

        if len(generated_df) > TOTAL_GENERATION_BUDGET:
            generated_df = generated_df.iloc[:TOTAL_GENERATION_BUDGET].copy()

        generated_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\n生成完成！正式输出 {len(generated_df)} 条合法 IPv6 地址")
        print(f"结果已保存到：{OUTPUT_FILE}")
        if len(generated_df) < TOTAL_GENERATION_BUDGET:
            print(f"警告：正式输出数量低于目标预算 {TOTAL_GENERATION_BUDGET}，可尝试提高 TOP_K、TEMPERATURE 或 MAX_ATTEMPTS_MULTIPLIER。")
    else:
        print("\n未生成任何合法 IPv6 地址！")


if __name__ == "__main__":
    main()
