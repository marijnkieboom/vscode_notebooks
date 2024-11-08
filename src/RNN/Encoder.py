import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """
        Args:
            input: [batch_size, seq_len] - Input sequence of token indices

        Returns:
            output: [batch_size, seq_len, 2 * hidden_size] - Encodings of the whole input
            hidden: [num_layers, batch_size, 2 * hidden_size] - Final hidden states of the encoder, which can be used to initialize the decoder hidden states

        The forward function of the Encoder is called once for the entire
        input sequence. The RNN steps as many times as the lenght of the
        sequence. This is done simultaneously for all batches.
        """

        batch_size = input.shape[0]  # batch_size

        # Get learnable multi-dimensional embeddings of the input matrix
        embeddings = self.dropout(self.embeddings(input)) # embeddings = [batch_size, seq_len, embedding_dim]

        # Run the embeddings through the bidirectional RNN
        output, hidden = self.rnn(embeddings)
        # output = [batch_size, seq_len, 2 * hidden_size]
        # hidden = [2 * num_layers, batch_size, hidden_size]

        # Concat hidden states from forward and backward passes to obtain the annotations """
        hidden = hidden.reshape(self.num_layers, 2, batch_size, self.hidden_size)
        # hidden = [num_layers, 2, batch_size, hidden_size]

        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        # hidden = [num_layers, batch_size, 2 * hidden_size]

        return output, hidden
