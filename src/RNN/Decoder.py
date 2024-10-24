import torch
import torch.nn as nn
import torch.nn.functional as F

from RNN.Attention import Attention


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.attention = Attention(num_layers, hidden_size)
        self.rnn = nn.GRU(
            emb_dim + hidden_size * 2,
            hidden_size,
            num_layers,
            batch_first=True,
            # dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encodings):
        """
        The forward function of the Decoder is called for every new token.
        The Decoder RNN only steps a single time frame forwards. The step
        is made simultaneously for all batches.
        """

        # input     = [N]
        # hidden    = [num layers, N, hidden size]
        # encodings = [N, L, 2 * hidden size]

        """Get learnable multi-dimensional embeddings of the input"""
        embeddings = self.dropout(self.embeddings(input))
        # embeddings = [N, embedding dim]

        """ Recompute context vector every time step, using the hidden state
        of the previous time step, and the encoder outputs """
        context_vector, att_weights = self.attention(hidden, encodings)
        # context_vector = [N, hidden size * 2]
        # att_weights    = [N, L]

        """ Input for the RNN is a combination of the embeddings of the last
            generated token, and weighted encodings """
        rnn_input = torch.cat((embeddings, context_vector), dim=1)
        # rnn_input = [N, embedding dim + 2 * hidden size]

        """ Add a single sequence length dimension to the RNN input"""
        rnn_input = rnn_input.unsqueeze(1)
        # rnn_input = [N, 1, embedding dim + 2 * hidden size]

        """ Iterate only one step """
        output, hidden = self.rnn(rnn_input, hidden)
        # output = [N, 1, hidden size]
        # hidden = [num layers, N, hidden size]

        """ Remove the single sequence length dimension again """
        output = output.squeeze(1)
        # output = [N, hidden size]

        """ Make a prediction for the next token based on the output """
        prediction = self.fc(output)
        # prediction = [N, vocab size]

        return prediction, hidden, att_weights
