import torch
from torch import nn
import torch.nn.functional as F

from config import LM_Config


config = LM_Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_1 = nn.Linear(config.d_model, config.d_hidden)
        self.W_2 = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x):
        x = F.gelu(self.W_1(x))
        x = self.W_2(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)

        self.W_proj = nn.Linear(config.d_model, config.d_model)

    def mask(self, score, always_attend_upto):
        batch_size, _, seq_len, _ = score.shape

        if config.use_prefix_lm_masking:
            causal = [
                torch.tril(torch.ones(seq_len, seq_len), diagonal=0) for _ in range(batch_size)
            ]

            causal_w_prefix = [torch.cat([
                torch.ones(seq_len, length),
                torch.zeros(seq_len, seq_len - length)
            ], dim=1) for length in always_attend_upto]

            mask = [(causal[i] + causal_w_prefix[i]) == 0 for i in range(batch_size)]
            mask = torch.stack(mask, dim=0).unsqueeze(1).to(device)

            return score.masked_fill(mask, float("-inf"))

        else:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=score.device),
                diagonal=1
            ).bool().to(device)
            return score.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    def forward(self, x, always_attend_upto):
        # could have calculated q-k-v in one single matmul too
        query = self.W_q(x).reshape(
            -1, config.context_length, config.num_heads, config.d_head
        ).transpose(1, 2)

        key = self.W_k(x).reshape(
            -1, config.context_length, config.num_heads, config.d_head
        ).transpose(1, 2)

        value = self.W_v(x).reshape(
            -1, config.context_length, config.num_heads, config.d_head
        ).transpose(1, 2)

        QK_t = torch.matmul(query, key.transpose(3, 2))

        d_k = query.shape[-1]
        score = QK_t / torch.sqrt(torch.tensor(d_k, dtype=torch.float, device=QK_t.device))
        masked_score = self.mask(score, always_attend_upto)
        masked_score = F.softmax(masked_score, dim=-1)

        value = torch.matmul(masked_score, value)
        value = value.transpose(1, 2).contiguous().reshape(
            -1, config.context_length, config.d_model
        )

        value = self.W_proj(value)  # this is different than the position-wise FFNs

        return value


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x, always_attend_upto):
        x = self.ln1(x)
        x = x + self.attention(x, always_attend_upto)
        x = self.ln2(x)
        x = x + self.ffn(x)

        return x


class LM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block() for _ in range(config.num_block)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.token_emb.weight)

    def forward(self, x, always_attend_upto):
        pos_x = torch.arange(0, config.context_length, dtype=torch.long).to(device)
        x = self.token_emb(x) + self.pos_emb(pos_x)

        for i in range(config.num_block):
            x = self.blocks[i](x, always_attend_upto)

        x = self.lm_head(x)
        x = F.log_softmax(x, dim=-1)

        return x
