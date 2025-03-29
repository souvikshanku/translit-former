import json

import pandas as pd
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, context_length, data, tokens):
        self.context_length = context_length
        self.words_df = pd.read_csv(data)
        self.tokens = json.load(open(tokens, "r"))

    def __len__(self):
        return len(self.words_df)

    def __getitem__(self, idx):
        bengali = str(self.words_df.iloc[idx]["bn"]).lower()
        english = str(self.words_df.iloc[idx]["en"]).lower()

        tokens = (
            [self.tokens[b] for b in bengali]
            + [self.tokens["</s>"]]
            + [self.tokens[e] for e in english]
        )

        pad = [self.tokens["</s>"]] * (self.context_length - len(tokens))

        mask = [0] * len(bengali) + [1] * (len(english) + 1)
        mask += [0] * (self.context_length - len(mask))

        length = len(bengali)  # from prefix-LM, attend the whole word
        return (tokens + pad), mask, length + 1
