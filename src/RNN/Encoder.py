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
            # dropout=dropout,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """
        The forward function of the Encoder is called once for the entire
        input sequence. The RNN steps as many times as the lenght of the
        sequence. This is done simultaneously for all batches.
        """

        # input = [N, L]

        batch_size = input.shape[0]  # N

        """ Get learnable multi-dimensional embeddings of the input matrix """
        embeddings = self.dropout(self.embeddings(input))
        # embeddings = [N, L, embedding dim]

        """ Run the embeddings through an RNN, iterating over the `sequence length` dimension """
        output, hidden = self.rnn(embeddings)
        # output = [N, L, 2 * hidden size]
        # hidden = [2 * num layers, N, hidden size]

        """  Concat hidden states of forward and backward passes to obtain the annotations """
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        # [num layers, 2, N, hidden size]
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        # [num layers, N, 2 * hidden size]

        """
         * Encoder outputs: Sequence of (the sum of) hidden states from the encoder for each time step.
         * Hidden state:    The last hidden state from the encoder, which can be used to initialize the decoder's hidden state.
        """
        return output, hidden
