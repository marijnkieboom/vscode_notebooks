from collections import OrderedDict

class Vocabulary:
    def __init__(self, ordered_dict: OrderedDict, min_freq):
        self.min_freq = min_freq
        self.specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

        filtered_dict = {token: freq for token, freq in ordered_dict.items() if freq >= self.min_freq}

        self.token_to_index = {token: idx for idx, token in enumerate(self.specials)}
        self.token_to_index.update({token: idx + len(self.specials) for idx, token in enumerate(filtered_dict.keys())})

        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}

        self.size = len(self.token_to_index)

    def token_to_index_func(self, token):
        return self.token_to_index.get(token.lower(), self.token_to_index['<UNK>'])

    def index_to_token_func(self, index):
        return self.index_to_token.get(index, None)

    def __len__(self):
        return self.size

    def __str__(self):
        return f"Vocabulary with {self.size} tokens."
