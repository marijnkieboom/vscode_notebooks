import json
import pandas as pd
from collections import Counter, OrderedDict

def main():
    chunksize = 500000
    tokenized_file = "./data-sets/filtered_and_tokenized_sentences(2).csv"
    vocab_file_eng = "./data-sets/vocab_eng.json"
    vocab_file_nld = "./data-sets/vocab_nld.json"

    eng_counter = Counter()
    nld_counter = Counter()

    for chunk in pd.read_csv(tokenized_file, chunksize=chunksize, header=None, names=["eng_tokens", "nld_tokens"]):
        for _, row in chunk.iterrows():
            eng_counter.update([token.lower() for token in row["eng_tokens"].split()])
            nld_counter.update([token.lower() for token in row["nld_tokens"].split()])

    env_vocab = OrderedDict(sorted(eng_counter.items(), key=lambda x: x[1], reverse=True))
    nld_vocab = OrderedDict(sorted(nld_counter.items(), key=lambda x: x[1], reverse=True))

    with open(vocab_file_eng, mode="w") as f:
        json.dump(env_vocab, f)

    with open(vocab_file_nld, mode="w") as f:
        json.dump(nld_vocab, f)

if __name__ == "__main__":
    main()
