import torch
from torch import nn
import torch.nn.functional as F

CONTEXT_LENGTH = 64
USE_PREFIX_LM_MASKING = True

device = torch.device("cpu")


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.d_model = 32
        self.d_hidden = 64

        self.W_1 = nn.Linear(self.d_model, self.d_hidden)
        self.W_2 = nn.Linear(self.d_hidden, self.d_model)

    def forward(self, x):
        x = F.gelu(self.W_1(x))
        x = self.W_2(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.context_length = CONTEXT_LENGTH
        self.d_model = 32
        self.num_heads = 4
        self.d_head = self.d_model // self.num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)

        self.c_proj = nn.Linear(self.d_model, self.d_model)

    # @staticmethod
    def mask(self, score, always_attend_upto):
        batch_size, _, seq_len, _ = score.shape

        if USE_PREFIX_LM_MASKING:
            causal = [
                torch.tril(torch.ones(seq_len, seq_len), diagonal=0) for _ in range(batch_size)
            ]

            causal_w_prefix = [torch.cat([
                torch.ones(seq_len, length),
                torch.zeros(seq_len, seq_len - length)
            ], 1) for length in always_attend_upto]

            mask = [(causal[i] + causal_w_prefix[i]) == 0 for i in range(batch_size)]
            mask = torch.stack(mask, dim=0).unsqueeze(1)

            return score.masked_fill(mask, float("-inf"))

        else:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=score.device), diagonal=1).bool()
            return score.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    def forward(self, x, always_attend_upto):
        # query = self.W_q(x).reshape(-1, self.num_heads, self.context_length, self.d_head)
        # key = self.W_k(x).reshape(-1, self.num_heads, self.context_length, self.d_head)
        # value = self.W_v(x).reshape(-1, self.num_heads, self.context_length, self.d_head)

        query = self.W_q(x).reshape(
            -1, self.context_length, self.num_heads, self.d_head
        ).transpose(1, 2)

        key = self.W_k(x).reshape(
            -1, self.context_length, self.num_heads, self.d_head
        ).transpose(1, 2)

        value = self.W_v(x).reshape(
            -1, self.context_length, self.num_heads, self.d_head
        ).transpose(1, 2)

        QK_t = torch.matmul(query, key.transpose(3, 2))

        d_k = query.shape[-1]
        score = QK_t / torch.sqrt(torch.tensor(d_k, dtype=torch.float, device=QK_t.device))
        masked_score = self.mask(score, always_attend_upto)
        masked_score = F.softmax(masked_score, dim=-1)

        value = torch.matmul(masked_score, value)
        value = value.transpose(1, 2).contiguous().reshape(-1, self.context_length, self.d_model)

        value = self.c_proj(value)

        return value


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        # self.num_block = 4
        self.context_length = CONTEXT_LENGTH
        self.d_model = 32

        self.attention = SelfAttention()
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x, always_attend_upto):
        x = self.ln1(x)
        x = x + self.attention(x, always_attend_upto)
        x = self.ln2(x)
        x = x + self.ffn(x)

        return x


class LM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_block = 3
        self.vocab_size = 163
        self.context_length = CONTEXT_LENGTH
        self.d_model = 32

        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=32)
        self.pos_emb = nn.Embedding(num_embeddings=CONTEXT_LENGTH, embedding_dim=32)
        self.blocks = nn.ModuleList([Block() for _ in range(self.num_block)])
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.token_emb.weight)

    def forward(self, x, always_attend_upto):
        pos_x = torch.arange(0, self.context_length, dtype=torch.long).to(device)
        x = self.token_emb(x) + self.pos_emb(pos_x)

        for i in range(self.num_block):
            x = self.blocks[i](x, always_attend_upto)

        x = self.lm_head(x)
        x = F.log_softmax(x, dim=-1)

        return x
