import torch
import torch.nn as nn
import torch.nn.functional as F

### Bahdanau Attention mechanism ###

""" The attention determines which part of the input (encodings) is relevant for the current decoder's hidden state """


class Attention(nn.Module):

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()

        self.W1 = nn.Linear(num_layers * hidden_size, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encodings):
        """
        The decoder's hidden state is also known as the 'Query', and the
        encoder's outputs as the 'Values'. The query changes with each
        time step, but the values remain the same for every step.
        """

        # hidden    = [num layers, N, hidden size]
        # encodings = [N, L, 2 * hidden size]

        hidden = torch.movedim(hidden, 0, 1)
        # hidden = [N, num layers, hidden size]

        batch_size = hidden.shape[0]  # N
        num_layers = hidden.shape[1]
        hidden_size = hidden.shape[2]

        concat_hidden = hidden.reshape(batch_size, num_layers * hidden_size)
        # concat_hidden = [N, num layers * hidden size]

        """ Add a dimension to match the encodings """
        concat_hidden = concat_hidden.unsqueeze(1)
        # concat_hidden = [N, 1, num layers * hidden size]

        """ Calculate scores """
        scores = self.V(torch.tanh(self.W1(concat_hidden) + self.W2(encodings)))
        # scores = [N, L, 1]

        scores = torch.movedim(scores, 1, 2)
        # scores = [N, 1, L]

        """ Rescale so that the scores lie in the range of [0-1] and sum to 1 """
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights = [N, 1, L]

        """ Calculate the context vector """
        context_vector = torch.bmm(attention_weights, encodings)
        # context_vector = [N, 1, 2 * hidden size]

        context_vector = context_vector.squeeze(1)
        attention_weights = attention_weights.squeeze(1)
        # context_vector = [N, 2 * hidden size]
        # attention_weights = [N, L]

        return context_vector, attention_weights
