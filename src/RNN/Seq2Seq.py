import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from RNN.Encoder import Encoder
from RNN.Decoder import Decoder


class Seq2Seq(nn.Module):

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        emb_dim: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        """Construct the encoder"""
        self.encoder = Encoder(
            input_vocab_size,
            emb_dim,
            encoder_hidden_size,
            num_layers,
            dropout,
        )

        """Construct the decoder"""
        self.decoder = Decoder(
            output_vocab_size,
            emb_dim,
            decoder_hidden_size,
            num_layers,
            dropout,
        )

        """Connect the encoder and decode with a fully connected layer"""
        self.fc_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

    def forward(self, source, target, forced_teaching_ratio=0):
        # source = [N, L]
        # target = [N, L]

        device = next(self.parameters()).device

        batch_size = target.shape[0]  # N
        sequence_length = target.shape[1]  # L
        target_vocab_size = self.decoder.vocab_size

        """ Encode the whole input sequence using the encoder """
        encodings, hidden = self.encoder(source)
        # encodings = [N, L, 2 * hidden size]
        # hidden    = [num layers, N, 2 * hidden size]

        """ Compute initial hidden state for the decoder based on the encoder's last hidden state """
        hidden = torch.tanh(self.fc_hidden(hidden))
        # hidden = [num layers, N, hidden size]

        """ Create a primer for the Decoder """
        input = target[:, 0]  # first input to the decoder is the <sos> tokens
        # input = [N]

        """ Create a tensor to save the results to """
        outputs = torch.zeros(sequence_length, batch_size, target_vocab_size).to(device)
        # outputs = [L, N, vocab size]

        """ Generate a single new token every iteration, parallel for all batches """
        for i in range(1, sequence_length):

            """Decode the next output in the sequence with the previously generated token as input"""
            output, hidden, _ = self.decoder(input, hidden, encodings)
            # output = [N, vocab size]
            # hidden = [num layers, N, 2 * hidden size]

            outputs[i] = output  # Save generated output in a new dimension

            """ Determine next input token for the decoder, possibly using forced teaching """
            if random.random() < forced_teaching_ratio:
                input = target[:, i]  # ground truth token
            else:
                input = torch.argmax(output, dim=1)

        outputs = outputs.permute(
            1, 2, 0
        )  # Shuffle dimensions around to match what the criterion wants
        # outputs = [N, vocab size, L]

        return outputs

    @torch.inference_mode()
    def evaluate(self, ixs, device, max_output_length=20):
        self.eval()

        tensor = torch.tensor(ixs).to(device)
        # tensor = [L]

        tensor = tensor.unsqueeze(0)  # Add a single batch dimension
        # tensor = [1, L]

        """ Encode whole input sequence using the encoder """
        encodings, hidden = self.encoder(tensor)
        # encodings = [1, L, 2 * hidden size]
        # hidden = [num layers, 1, 2 * hidden size]

        """ Compute initial hidden state for the decoder based on the encoder's last hidden state """
        hidden = torch.tanh(self.fc_hidden(hidden))

        """ Create a primer for the Decoder """
        indices = [1]  # first input to the decoder is the <sos> tokens

        attentions = []
        for _ in range(max_output_length):
            """Decode with the previous generated index as input"""
            inputs_tensor = torch.tensor(indices[-1]).unsqueeze(0).to(device)
            output, hidden, attention = self.decoder(inputs_tensor, hidden, encodings)
            # output =    [1, vocab size]
            # hidden =    [num layers, 1, 2 * hidden size]
            # attention = [1, L]

            output = output.squeeze(0)
            # output = [vocab size]

            prediction = torch.argmax(output).item()

            indices.append(prediction)
            attentions.append(attention)

            """ Stop generating when End Of Sequence token is generated """
            if prediction == 2:
                break

        attentions_tensor = torch.cat(attentions)
        # attentions_tensor = [L (out) - 1, L (in)]

        return indices, attentions_tensor
