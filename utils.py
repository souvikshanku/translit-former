import torch

from model import LM


def tokenize(text, tokens):
    text = text.strip().lower()
    ids = [tokens[k] for k in list(text)]
    ids = ids + [tokens["</s>"]] * (64 - len(ids))
    return text, ids


def transliterate(text: str, lm: LM, tokens) -> str:
    tokens_rev = {v: k for k, v in tokens.items()}
    device = next(lm.parameters()).device

    next_token_id = None

    text, token_ids = tokenize(text, tokens)
    always_attend_upto = [len(text)]
    length = len(text)

    while next_token_id != tokens["</s>"] and length <= 62:
        x = torch.tensor(token_ids)

        logits = lm(x.to(device), always_attend_upto)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        next_token_id = preds[0][length].item()

        token_ids[length + 1] = next_token_id
        length += 1

    return "".join([tokens_rev[k] for k in token_ids]).split("</s>")[1]
