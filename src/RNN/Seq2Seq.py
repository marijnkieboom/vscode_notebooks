import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from RNN.Encoder import Encoder as Bidirection_Encoder
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

        """ Construct a Bidirectional encoder """
        self.encoder = Bidirection_Encoder(
            input_vocab_size,
            emb_dim,
            encoder_hidden_size,
            num_layers,
            dropout,
        )

        """ Construct the decoder """
        self.decoder = Decoder(
            output_vocab_size,
            emb_dim,
            decoder_hidden_size,
            num_layers,
            dropout,
        )

        """ Construct a Linear Layer (also known as a fully connected layer) to map data into a lower-
        dimensional space. The encoder's final hidden state will be transformed into the initial hidden
        state of the decoder. """
        self.fc_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

    def forward(self, source, target, forced_teaching_ratio=0):
        batch_size = source.shape[0]
        sequence_length = source.shape[1]

        encodings, hidden = self.encoder(source) # Encode the input whole
        hidden = torch.tanh(self.fc_hidden(hidden))

        input = torch.zeros(batch_size, dtype=torch.long).to(source.device)
        input[:] = 1 # Primer for the decoder is an <SOS> token (ix = 1)
        outputs = []

        # Run the decoder 1 step at at time
        for i in range(sequence_length):
            output, hidden, _ = self.decoder(input, hidden, encodings)
            outputs.append(output.unsqueeze(0))

            use_teacher_forcing = torch.rand(1).item() < forced_teaching_ratio
            if use_teacher_forcing:
                input = target[:, i] # Ground truth token
            else:
                input = torch.argmax(output, dim=1)

        outputs = torch.cat(outputs, dim=0).permute(1, 2, 0) # [batch_size, vocab_size, seq_len]
        return outputs

    @torch.inference_mode()
    def evaluate(self, ixs, device, max_output_length=20):
        self.eval()

        input = torch.tensor(ixs, device=device).unsqueeze(0) # tensor: [1, seq_len]

        encodings, hidden = self.encoder(input)
        hidden = torch.tanh(self.fc_hidden(hidden)) # Create decoder hidden state

        input = torch.tensor([1], device=device) # Primer is an <SOS> tokens

        output_ixs = []
        attentions = []

        for _ in range(max_output_length):
            output, hidden, attention = self.decoder(input, hidden, encodings)
            attentions.append(attention)

            predicted_token_ix = torch.argmax(output, dim=1).item()
            output_ixs.append(predicted_token_ix)

            input = torch.tensor([predicted_token_ix], device=device)

            # Stop generating when End Of Sequence token is generated
            if predicted_token_ix == 2: # <EOS>
                break

        # Concatenate along the time dimension
        attentions = torch.cat(attentions, dim=0)

        return output_ixs, attentions
