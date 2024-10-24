MIN_count = 4

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class Vocabulary:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0

        [self.add_word(word) for word in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]]

    def add_word(self, word: str):
        if word not in self.word2index:
            ix = self.num_words

            self.word2index[word] = ix
            self.index2word[ix] = word
            self.word2count[word] = 1

            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def trim(self):
        keep_words = []

        for k, v in self.word2count.items():
            if v >= MIN_count:
                keep_words.append(k)

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0

        [self.add_word(word) for word in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]]

        for word in keep_words:
            self.add_word(word)

    def to_word(self, index: int) -> str:
        return self.index2word[index]

    def to_index(self, word: str) -> int:
        if word in self.word2index:
            return self.word2index[word]
        else:
            return UNK_token

    def lookup_indices(self, tokens: list) -> list:
        return [self.to_index(token) for token in tokens]

    def lookup_tokens(self, indices: list) -> list:
        return [self.to_word(index) for index in indices]

    def __len__(self):
        return self.num_words
