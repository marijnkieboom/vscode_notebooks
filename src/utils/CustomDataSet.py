import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, file_path, output_size, input_vocab, target_vocab, nrows):
        self.data = []
        self.output_size = output_size
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

        chunksize=100000
        columns = ["eng_tokens", "nld_tokens"]
        dtype = { "eng_tokens": "str", "nld_tokens": "str" }

        for chunk in pd.read_csv(
            file_path,
            encoding="utf-8",
            chunksize=chunksize,
            header=None,
            names=columns,
            dtype=dtype,
            nrows=nrows
        ):
            self.data.extend(self.process_chunk(chunk))

    def process_chunk(self, df):
        result = []

        for _, row in df.iterrows():
            eng_idx = [self.input_vocab.token_to_index_func(token) for token in row["eng_tokens"].split()]
            nld_idx = [self.target_vocab.token_to_index_func(token) for token in row["nld_tokens"].split()]

            eng_idx = self.pad_or_truncate(eng_idx, self.input_vocab.token_to_index_func("<PAD>"))
            nld_idx = self.pad_or_truncate(nld_idx, self.target_vocab.token_to_index_func("<PAD>"))

            result.append((torch.tensor(eng_idx, dtype=torch.int), torch.tensor(nld_idx, dtype=torch.int)))

        return result

    def pad_or_truncate(self, tokens, pad_idx):
        if (len(tokens) == self.output_size):
            return tokens

        if len(tokens) > self.output_size:
            return tokens[:self.output_size]

        return tokens + [pad_idx] * (self.output_size - len(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
