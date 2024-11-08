import torch
import torch.nn as nn
import torch.nn.functional as F

### Bahdanau Attention mechanism ###

""" The attention determines which part of the input (encodings) is relevant for the current decoder's hidden state """


class Attention(nn.Module):

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()

        self.W1 = nn.Linear(num_layers * hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encodings):
        """
        The decoder's hidden state is also known as the 'Query', and the
        encoder's outputs as the 'Values'. The query changes with each
        time step, but the values remain the same for every step.
        """

        # hidden: [num_layers, batch_size, hidden_size]
        # encodings: [batch_size, seq_len, 2 * hidden_size]

        batch_size = hidden.shape[1]
        num_layers = hidden.shape[0]
        hidden_size = hidden.shape[2]

        # Concatenate hidden states of all layers
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)  # [batch_size, num_layers * hidden_size]

        # Expand hidden state for broadcasting
        hidden_exp = self.W1(hidden).unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Calculate attention scores
        scores = self.V(torch.tanh(hidden_exp + self.W2(encodings)))  # [batch_size, seq_len, 1]

        # Rescale to [0, 1] and sum to 1
        attention_weights = F.softmax(scores.squeeze(-1), dim=-1)  # [batch_size, seq_len]

        # Calculate context vector as the weighted sum of encodings
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encodings)  # [batch_size, 1, 2 * hidden_size]

        return context_vector.squeeze(1), attention_weights
