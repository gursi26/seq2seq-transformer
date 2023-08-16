from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import string
import torch


class EngSpaDataset(Dataset):

    def __init__(self, csv_path, start_idx = 0, end_idx = None):
        self.df = self.read_df(csv_path, start_idx, end_idx)
        self.eng2idx, self.spa2idx, self.idx2spa = self.get_idx_mappings(self.df)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        eng, spa = self.df.iloc[index]
        eng, spa = self.add_tokens(eng, spa)
        eng, spa = self.embed(eng, spa)
        return eng, spa[:-1],  spa[1:]

    def preprocess(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def read_df(self, csv_path, start_idx, end_idx):
        if end_idx is not None:
            df = pd.read_csv(csv_path).iloc[start_idx:end_idx]
        else:
            df = pd.read_csv(csv_path).iloc[start_idx:]
        return df

    def get_idx_mappings(self, df):
        spa2idx = {word: i + 3 for i, word in enumerate(sorted(list(set(" ".join(list(np.array(df[["SPA"]]).reshape(-1))).split()))))}
        spa2idx["<PAD>"] = 0
        spa2idx["<SOS>"] = 1
        spa2idx["<EOS>"] = 2
        idx2spa = {i: word for word, i in spa2idx.items()}

        eng2idx = {word: i + 2 for i, word in enumerate(sorted(list(set(" ".join(list(np.array(df[["ENG"]]).reshape(-1))).split()))))}
        eng2idx["<PAD>"] = 0
        eng2idx["<UNK>"] = 1
        idx2spa = {i: word for word, i in spa2idx.items()}
        return eng2idx, spa2idx, idx2spa


    def add_tokens(self, eng, spa):
        eng, spa = eng.split(), ["<SOS>"] + spa.split() + ["<EOS>"]
        return eng, spa

    def generate_mask(self, eng_len, spa_len):
        eng_mask = torch.cat([torch.ones(eng_len), torch.zeros(self.eng_max_len - eng_len)]) == 1
        spa_mask = torch.cat([torch.ones(spa_len), torch.zeros(self.spa_max_len - spa_len)]) == 1
        return eng_mask, spa_mask

    def embed(self, eng, spa):
        eng = torch.tensor([self.eng2idx[word] for word in eng])
        spa = torch.tensor([self.spa2idx[word] for word in spa])
        return eng, spa


def test_dataset():
    d = EngSpaDataset("eng-spa.csv")
    print(len(d.spa2idx), len(d.eng2idx))
    # loader = DataLoader(d, batch_size=32)
    encoder_input, decoder_input, target = d[9830]
    print(f"Encoder input shape: {encoder_input.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Target shape: {target.shape}")

if __name__ == "__main__":
    test_dataset()