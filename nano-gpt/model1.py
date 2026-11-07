import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 64          # 每批多少条序列
block_size = 256         # 最大上下文长度
max_iters = 5000         # 训练迭代次数
eval_interval = 500      # 每多少步评估一次
learning_rate = 3e-4     # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200         # 评估用多少个 batch
n_embd = 384             # 隐藏维度 C
n_head = 6               # 注意力头数
n_layer = 6              # Transformer block 层数
dropout = 0.2            # dropout 比例
# ------------

torch.manual_seed(1337)

# 读取训练语料
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 字符级 tokenizer：统计所有字符，建立字符和 ID 的映射
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 词表大小 V
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符 -> ID
itos = { i:ch for i,ch in enumerate(chars) }  # ID -> 字符
encode = lambda s: [stoi[c] for c in s]  # 文本 -> ID
decode = lambda l: ''.join([itos[i] for i in l])  # ID -> 文本

# 划分训练集和验证集：前 90% 训练，后 10% 验证
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# 取一个 batch：x 是输入，y 是 x 向右移动一位后的目标
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # 关闭 dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # 恢复训练模式
    return out

class Head(nn.Module):
    """单个自注意力头。"""

    def __init__(self, head_size):
        super().__init__()
        # Q、K、V 投影
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 因果 mask：只能看当前和之前的 token
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)    # (B,T,hs)
        q = self.query(x)  # (B,T,hs)

        # 注意力分数，并做 causal mask
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 用注意力权重对 V 加权求和
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """多头注意力：多个 head 并行，再拼接。"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # 输出投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """前馈网络：对每个 token 单独做特征变换。"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 升维
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 降回原维度
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """一个 Transformer block：attention + FFN，带残差。"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LN + 残差
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # token embedding 和 position embedding
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # 输出 V 个 logits

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # token + 位置信息
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        if targets is None:
            loss = None  # 只生成，不算 loss
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  # 训练 loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # 逐 token 生成
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 只保留最后 block_size 个 token
            logits, loss = self.forward(idx_cond)  # 前向
            logits = logits[:, -1, :]  # 只取最后一个位置
            probs = F.softmax(logits, dim=-1)  # 变成概率
            idx_next = torch.multinomial(probs, num_samples=1)  # 采样一个 token
            idx = torch.cat((idx, idx_next), dim=1)  # 拼到末尾
        return idx

# 创建模型
model = GPTLanguageModel()

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练循环
for iter in range(max_iters):

    # 定期评估
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 取 batch，前向、反向、更新
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 训练完生成
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
