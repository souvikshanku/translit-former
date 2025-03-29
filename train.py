import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import json

from dataloader import Data
from model import LM


if __name__ == "__main__":
    with open("tokenizer.json", "r") as f:
        tokens = json.load(f)

    tokens_rev = {v: k for k, v in tokens.items()}

    words_dict = json.load(open("data.json", "r"))

    with open("test.json", "r") as f:
        test_json = json.load(f)

    with open("train.json", "r") as f:
        train_json = json.load(f)

    def collate_fn(batch):
        return batch  # otherwise list[list] somehow gets converted to list[tensor]

    CONTEXT_LENGTH = 64

    dataset = Data(CONTEXT_LENGTH, "train.json", "tokenizer.json")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cpu")

    lm = LM()
    lm.to(device)

    lm.train()

    optimizer = optim.Adam(lm.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    lm.train()

    num_epochs = 50

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_id, data in enumerate(dataloader):
            x = torch.tensor([i[0] for i in data]).to(device)
            logit_mask = torch.tensor([i[1] for i in data]).to(device)
            always_attend_upto = [i[2] for i in data]

            logits = lm(x, always_attend_upto)
            masked_logits = (logits * logit_mask.unsqueeze(dim=-1))

            loss = - (masked_logits[:, :-1, :] * F.one_hot(x[:, 1:], num_classes=140)).sum() / logit_mask.sum()  # noqa: E501

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}: Avg loss: {epoch_loss / (batch_id + 1)}")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Avg loss: {epoch_loss / (batch_id + 1)}")

            torch.save({
                "epoch": epoch + 1,
                "lm_state_dict": lm.state_dict(),
            }, f"checkpoints/epoch_{epoch + 1}.pt")
            print("Checkpoint saved...")
