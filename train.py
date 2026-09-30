import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# -------------------------- 1. 设备配置 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. 加载预处理数据 --------------------------
# 适配17段输入/16段输出
src_sequences = np.load("src_sequences.npy", allow_pickle=True).astype(int)  # (N, 17)
tgt_sequences = np.load("tgt_sequences.npy", allow_pickle=True).astype(int)  # (N, 16)

# 加载词表
with open("token2id.json", "r") as f:
    token2id = json.load(f)
with open("id2token.json", "r") as f:
    id2token = json.load(f)
vocab_size = len(token2id)
pad_token_id = token2id["<PAD>"]
unk_token_id = token2id["<UNK>"]

# -------------------------- 3. 数据集类（适配17/16段序列） --------------------------
class IPv6Dataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs, pad_token_id):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs
        self.pad_token_id = pad_token_id
        # 统一序列长度（17段输入，16段输出）
        self.src_max_len = 17
        self.tgt_max_len = 16

    def __len__(self):
        return len(self.src_seqs)

    def pad_sequence(self, seq, max_len):
        """填充序列至指定长度"""
        if len(seq) < max_len:
            seq = seq + [self.pad_token_id] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

    def __getitem__(self, idx):
        src = self.pad_sequence(self.src_seqs[idx], self.src_max_len)
        tgt = self.pad_sequence(self.tgt_seqs[idx], self.tgt_max_len)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# -------------------------- 4. 模型定义（6层编码器/解码器） --------------------------
class TransformerIPGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 嵌入层（共享词表，padding_idx=pad_token_id）
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        # 6层Transformer编码器/解码器
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,  # 6层
            num_decoder_layers=num_decoder_layers,  # 6层
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 适配[batch, seq_len, dim]格式
        )
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_positional_encoding(self, seq_len):
        """生成位置编码（适配17/16段序列长度）"""
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-np.log(10000.0) / self.d_model))
        pos_enc = torch.zeros(1, seq_len, self.d_model, device=device)
        pos_enc[0, :, 0::2] = torch.sin(pos * div_term)
        pos_enc[0, :, 1::2] = torch.cos(pos * div_term)
        return pos_enc

    def generate_masks(self, src, tgt):
        """生成Transformer掩码（适配17/16段序列）"""
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        
        # 源序列PAD掩码 [batch_size, src_seq_len]
        src_key_padding_mask = (src == self.embedding.padding_idx).to(device)
        # 目标序列PAD掩码 [batch_size, tgt_seq_len]
        tgt_key_padding_mask = (tgt == self.embedding.padding_idx).to(device)
        # 目标序列Look-Ahead掩码 [tgt_seq_len, tgt_seq_len]
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()
        
        return None, tgt_mask, src_key_padding_mask, tgt_key_padding_mask

    def forward(self, src, tgt):
        """前向传播：适配17段输入，16段输出"""
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        
        # 嵌入+位置编码
        src_emb = self.dropout(self.embedding(src) + self.generate_positional_encoding(src_seq_len))
        tgt_emb = self.dropout(self.embedding(tgt) + self.generate_positional_encoding(tgt_seq_len))
        
        # 生成掩码
        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.generate_masks(src, tgt)
        
        # Transformer前向计算
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 输出层
        out = self.fc_out(out)
        return out

# -------------------------- 5. 训练配置 --------------------------
# 数据集加载
dataset = IPv6Dataset(src_sequences, tgt_sequences, pad_token_id)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# 模型初始化（6层编码器/解码器）
model = TransformerIPGenerator(
    vocab_size=vocab_size,
    d_model=128,
    nhead=8,
    num_encoder_layers=6,  
    num_decoder_layers=6,  
    dim_feedforward=512,
    dropout=0.1
).to(device)

# 损失函数（忽略PAD）
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
# 优化器（AdamW带权重衰减，抑制过拟合）
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# 学习率调度器（余弦退火，适配6层模型）
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# 训练参数
epochs = 100
best_loss = float("inf")

# -------------------------- 6. 训练循环 --------------------------
print(f"开始训练（设备：{device}），总轮次：{epochs}，批次大小：64，模型层数：6层编码器/解码器")
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
        src_batch = src_batch.to(device)  # (batch, 17)
        tgt_batch = tgt_batch.to(device)  # (batch, 16)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播（tgt输入取前n-1段，输出预测后n-1段）
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        output = model(src_batch, tgt_input)
        
        # 计算损失（reshape为[batch*seq_len, vocab_size]）
        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt_output.reshape(-1)
        )
        
        # 反向传播+优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印批次进度
        if (batch_idx + 1) % 10 == 0:
            avg_batch_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {avg_batch_loss:.4f}")
    
    # 学习率调度
    scheduler.step()
    
    # 计算本轮平均损失
    avg_epoch_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] 完成，平均损失：{avg_epoch_loss:.4f}，当前学习率：{scheduler.get_last_lr()[0]:.6f}")
    
    # 保存最优模型
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), "transformer_ip_generator.pth")
        print(f"  保存最优模型（损失：{best_loss:.4f}）")

# -------------------------- 7. 训练完成 --------------------------
print(f"\n训练结束！最优模型已保存为 transformer_ip_generator.pth，最优损失：{best_loss:.4f}")