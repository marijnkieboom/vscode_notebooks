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

        self.input_fc = nn.Linear(emb_dim + hidden_size * 2, hidden_size)

        self.rnn = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encodings):
        """
        Args:
            input: [batch_size] - Token indices for the current time step.
            hidden: [num_layers, batch_size, hidden_size] - Hidden states from the previous time step.
            encodings: [batch_size, seq_len, 2 * hidden_size] - Encoder outputs (used for attention).

        Returns:
            prediction: [batch_size, vocab_size] - Predicted next token distribution.
            hidden: [num_layers, batch_size, hidden_size] - Updated hidden state for the next time step.
            att_weights: [batch_size, seq_len] - Attention weights for the encoder outputs.
        """
        # Get learnable multi-dimensional embeddings of the input token
        embeddings = self.dropout(self.embeddings(input)) # embeddings: [batch_size, embedding_dim]

        # Compute context vector using the current hidden state (query) and encoder outputs (values)
        context_vector, att_weights = self.attention(hidden, encodings)
        # context_vector: [batch_size, 2 * hidden_size]
        # att_weights: [batch_size, seq_len] - for visualization purposes only

        # Prepare the input for the RNN by concatenating the embeddings and the context vector
        combined_input = torch.cat((embeddings, context_vector), dim=1)
        # combined_input: [batch_size, embedding_dim + 2 * hidden_size]

        # Apply the linear layer to combine the embeddings and context vector in a learnable fasion
        rnn_input = torch.tanh(self.input_fc(combined_input))
        # rnn_input: [batch_size, hidden_size]

        # Add a single time step dimension
        rnn_input = rnn_input.unsqueeze(1)
        # rnn_input: [batch_size, 1, hidden_size]

        # Run the RNN for a single time step
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [batch_size, 1, hidden size]
        # hidden: [num layers, batch_size, hidden size]

        # Remove the sequence length dimension
        output = output.squeeze(1)
        # output: [batch_size, hidden size]

        # Generate predictions for the next token
        prediction = self.fc(output)
        # prediction: [batch_size, vocab size]

        return prediction, hidden, att_weights
