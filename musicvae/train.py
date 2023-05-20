import time
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from musicvae import layers, metrics, music_utils


def train_model(
    train_dataset: str, valid_dataset: str, device_name: str = "cpu", epochs: int = 150
) -> Tuple[nn.Module]:
    """Train the VAE model and return the encoder, conductor, decoder model as a tuple.

    :param train_dataset: Path to train dataset.
    :param valid_dataset: Path to validation dataset.
    :param device_name: Which device to use, defaults to "cpu".
    :param epochs: Total epochs to train, defaults to 50.
    :return: Trained models.
    """
    device = torch.device(device_name)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # input and output dimension is the total length of the one hot encoded of the pitch numbers
    pitch_one_hot_dim = 2 ** len(music_utils.PITCH_ENCODED)
    latent_dim_size = 512
    batch_size = 64

    # Create a Dataset and DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(np.load(train_dataset)))
    valid_dataset = TensorDataset(torch.from_numpy(np.load(valid_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    encoder = layers.MusicEncoder(
        input_size=pitch_one_hot_dim,
        hidden_size=2048,
        proj_size=512,
        latent_dim=latent_dim_size,
    ).to(device)
    conductor = layers.Conductor(
        input_size=latent_dim_size, hidden_size=1024, proj_size=512
    ).to(device)
    decoder = layers.MusicDecoder(
        input_size=512, hidden_size=1024, pitch_dim=pitch_one_hot_dim
    ).to(device)

    # Initialize optimizer and schedular for each model
    enc_optimizer = optim.Adam(encoder.parameters(), lr=1e-2)
    enc_schedular = optim.lr_scheduler.LambdaLR(
        enc_optimizer, lr_lambda=lambda epoch: 0.99**epoch
    )

    con_optimizer = optim.Adam(conductor.parameters(), lr=1e-2)
    con_schedular = optim.lr_scheduler.LambdaLR(
        con_optimizer, lr_lambda=lambda epoch: 0.99**epoch
    )

    dec_optimizer = optim.Adam(decoder.parameters(), lr=1e-2)
    dec_schedular = optim.lr_scheduler.LambdaLR(
        dec_optimizer, lr_lambda=lambda epoch: 0.99**epoch
    )

    # train
    for epoch_idx in range(1, epochs + 1):
        start_time = time.time()

        train_loss = 0
        train_acc = 0

        val_loss = 0
        val_acc = 0

        encoder.train()
        conductor.train()
        decoder.train()

        # iterate through each train dataset
        for batch_idx, x_train in enumerate(train_dataloader):
            x_train = x_train[0].to(dtype=torch.float32, device=device)

            enc_optimizer.zero_grad()
            con_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            train_z, train_mu, train_logvar = encoder(x_train)
            train_bar_feature = conductor(train_z)
            train_music_notes_prob = decoder(train_bar_feature)
            _, idx = torch.max(train_music_notes_prob, dim=2)
            mask = torch.arange(train_music_notes_prob.size(2)).reshape(
                1, 1, -1
            ) == idx.unsqueeze(2)
            train_music_notes_pred = torch.zeros_like(train_music_notes_prob)
            train_music_notes_pred[mask] = 1.0

            loss = metrics.ELBO_loss(
                train_music_notes_prob, x_train, train_mu, train_logvar
            )

            loss.backward()
            enc_optimizer.step()
            con_optimizer.step()
            dec_optimizer.step()

            train_loss += loss.item()
            train_acc += metrics.accuracy(x_train, train_music_notes_pred).item()

        enc_schedular.step()
        con_schedular.step()
        dec_schedular.step()

        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # validate
        encoder.eval()
        conductor.eval()
        decoder.eval()
        with torch.no_grad():
            for batch_idx, x_val in enumerate(valid_dataloader):
                x_val = x_val[0].to(dtype=torch.float32, device=device)

                val_z, val_mu, val_logvar = encoder(x_val)
                val_bar_feature = conductor(val_z)
                val_music_notes_prob = decoder(val_bar_feature)
                _, idx = torch.max(val_music_notes_prob, dim=2)
                mask = torch.arange(val_music_notes_prob.size(2)).reshape(
                    1, 1, -1
                ) == idx.unsqueeze(2)
                val_music_notes_pred = torch.zeros_like(val_music_notes_prob)
                val_music_notes_pred[mask] = 1.0

            loss = metrics.ELBO_loss(val_music_notes_prob, x_val, val_mu, val_logvar)

            val_loss += loss.item()
            val_acc += metrics.accuracy(x_val, val_music_notes_pred).item()

            val_loss /= batch_idx + 1
            val_acc /= batch_idx + 1
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if epoch_idx % 5 == 0:
            print(
                f"Epoch : {epoch_idx} ({time.time()-start_time:.2f}s) - train_loss : {train_loss:.2f} train_acc : {train_acc:.2f} val_loss : {val_loss:.2f} val_acc : {val_acc:.2f}"
            )

    return encoder, conductor, decoder
