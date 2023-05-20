import torch
from torch import nn


class MusicEncoder(nn.Module):
    """Encoder model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        latent_dim: int,
        *args,
        **kwargs
    ) -> None:
        """Constructor of the encoder.

        :param input_size: Input feature size.
        :param hidden_size: Hidden feature size.
        :param proj_size: Projection size.
        :param latent_dim: Latent vector dimension size.
        """
        super().__init__(*args, **kwargs)

        self.enc_output_size = proj_size * 2

        # the encoder is a bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            bidirectional=True,
            batch_first=True,
        )

        # get a latent vector from the LSTM
        self.mu = nn.Linear(self.enc_output_size, latent_dim)
        self.log_var = nn.Linear(self.enc_output_size, latent_dim)
        self.soft_plus = nn.Softplus()
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def forward(self, x):
        x, (h, _) = self.encoder(x)
        h = h.transpose(0, 1).reshape(-1, self.enc_output_size)

        # This process outputs a z vector sampled from a Gaussian distribution
        # which mean is `mu` and which std is `sigma`
        mu = self.mu(h)
        mu = self.norm(mu)
        log_var = self.log_var(h)
        log_var = self.soft_plus(log_var)

        sigma = torch.exp(log_var * 2)
        with torch.no_grad():
            epsilon = torch.randn_like(sigma)

        z = mu + (epsilon * sigma)

        return z, mu, log_var


class Conductor(nn.Module):
    """Conductor model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        bar_count: int = 4,
        *args,
        **kwargs
    ) -> None:
        """Constructor for the conductor.

        :param input_size: Input size from the decoder(A sample vector size).
        :param hidden_size: Hidden feature size.
        :param proj_size: Projection size.
        :param bar_count: How many bars to conduct, defaults to 4.
        """
        super().__init__(*args, **kwargs)

        self.bar_count = bar_count

        self.input_size = input_size
        self.proj_size = proj_size

        self.norm = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        # a conductor is a 2 layer single direction LSTM
        self.conductor = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=2,
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        bar_feature = torch.zeros(x.shape[0], self.bar_count, self.proj_size)

        # If there is N bars to conduct,
        # N iteration of the LSTM model will be done
        z_input = x
        for bar_idx in range(self.bar_count):
            # the first cell's initial h and c state is zero
            # but after that, the h and c will be inputted from the before LSTM cell
            output, (h, c) = (
                self.conductor(z_input)
                if bar_idx == 0
                else self.conductor(z_input, (h, c))
            )
            bar_feature[:, bar_idx, :] = output.squeeze()
            z_input = x

        return bar_feature


class MusicDecoder(nn.Module):
    """Decoder model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        pitch_dim: int,
        notes_per_bar: int = 16,
        *args,
        **kwargs
    ) -> None:
        """Constructor for the decoder.

        :param input_size: Input size of the tensor from the conductor.
        :param hidden_size: Hidden feature size.
        :param pitch_dim: Output dimension size(Usually same as the Encoder input size).
        :param notes_per_bar: How many notes exists in one bar, defaults to 16.
        """
        super().__init__(*args, **kwargs)

        self.notes_per_bar = notes_per_bar
        self.pitch_dim = pitch_dim

        # a decoder is a 2 layer single direction LSTM
        self.decoder = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=pitch_dim,
            num_layers=2,
        )
        self.logits = nn.Linear()
        # Use softmax for one hot encoding prediction
        self.softmax = nn.Softmax(2)

    def forward(self, x):
        music_notes = torch.zeros(
            x.shape[0], x.shape[1] * self.notes_per_bar, self.pitch_dim
        )

        # Note : This implementation may not be right
        # the main point is that
        # after getting N bar representation of each output from the conductor,
        # get a note feature from each bar feature
        
        # bar feat
        #    |
        #  note_1 --- note_2 --- note_3 --- note_4 --- ... --- note_k
        for bar_idx in range(x.shape[1]):
            input_ = x[:, bar_idx, :].unsqueeze(1)

            for note_idx in range(self.notes_per_bar):
                output, (h, c) = (
                    self.decoder(input_)
                    if note_idx == 0
                    else self.decoder(input_, (h, c))
                )

                music_notes[
                    :, bar_idx * self.notes_per_bar + note_idx, :
                ] = output.squeeze()
                input_ = torch.zeros_like(input_)

        music_notes = self.softmax(music_notes)

        return music_notes
