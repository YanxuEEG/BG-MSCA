

"""BG-MSCA model with MSGC, MSCA, BGMSCA, and Decoder modules.

This version preserves the layer parameters and forward computation of
the user's original code while using names that describe each module.
"""

import torch
from torch import nn


class Decoder(nn.Module):
    """Reconstruct a one-channel signal from 128-channel features."""

    def __init__(self):
        super().__init__()

        self.conv3_1 = nn.Conv1d(
            in_channels=128,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.relu1 = nn.ReLU()

        self.conv3_2 = nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()

        self.output_conv = nn.Conv1d(
            in_channels=16,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, features):
        features = self.relu1(self.conv3_1(features))
        features = self.relu2(self.conv3_2(features))
        return self.output_conv(features)


class MSGC(nn.Module):
    """Multi-Scale Gated Convolution using the original four branches."""

    def __init__(self):
        super().__init__()

        self.branch_k1 = nn.Sequential(
            nn.Conv1d(128, 128 // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128 // 4, 32, kernel_size=1),
            nn.Sigmoid(),
        )

        self.branch_k5 = nn.Sequential(
            nn.Conv1d(128, 128 // 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128 // 4, 32, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

        self.branch_k9 = nn.Sequential(
            nn.Conv1d(128, 128 // 4, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(128 // 4, 32, kernel_size=9, padding=4),
            nn.Sigmoid(),
        )

        self.branch_k15 = nn.Sequential(
            nn.Conv1d(128, 128 // 4, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(128 // 4, 32, kernel_size=15, padding=7),
            nn.Sigmoid(),
        )

    def forward(self, bigru_features):
        mask_k1 = self.branch_k1(bigru_features)
        mask_k5 = self.branch_k5(bigru_features)
        mask_k9 = self.branch_k9(bigru_features)
        mask_k15 = self.branch_k15(bigru_features)

        gating_mask = torch.cat(
            (mask_k1, mask_k5, mask_k9, mask_k15),
            dim=1,
        )

        # Retains the original element-wise gating operation.
        gated_local_features = gating_mask * bigru_features
        return gated_local_features


class MSCA(nn.Module):
    """Multi-Scale Cross Attention."""

    def __init__(self):
        super().__init__()

        self.msgc = MSGC()

        self.global_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
        )

        self.local_norm = nn.LayerNorm(128)
        self.global_norm = nn.LayerNorm(128)
        self.cross_norm = nn.LayerNorm(128)

    def forward(self, bigru_sequence):
        # MultiheadAttention: (batch, sequence_length, 128)
        # Conv1d: (batch, 128, sequence_length)
        channel_first_features = bigru_sequence.transpose(1, 2)

        gated_local_features = self.msgc(channel_first_features)

        # Retains the original local residual connection.
        local_features = (
            channel_first_features + gated_local_features
        ).transpose(1, 2)
        local_features = self.local_norm(local_features)

        global_features, _ = self.global_attention(
            bigru_sequence,
            bigru_sequence,
            bigru_sequence,
        )

        # Retains the original global residual connection.
        global_features = bigru_sequence + global_features
        global_features = self.global_norm(global_features)

        # Local features are Q; global features are K and V.
        cross_features, _ = self.cross_attention(
            local_features,
            global_features,
            global_features,
        )

        # Retains the original cross-attention residual connection.
        fused_features = global_features + cross_features
        fused_features = self.cross_norm(fused_features)
        return fused_features


class BGMSCA(nn.Module):
    """Bidirectional GRU with Multi-Scale Cross Attention."""

    def __init__(self, input_dim, hidden_dim, channels, seq_length):
        super().__init__()

        # Retained from the original constructor.
        self.seq_length = seq_length
        self.channels = channels

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.msca = MSCA()
        self.output_relu = nn.ReLU()
        self.decoder = Decoder()

    def forward(self, contaminated_signal):
        # (batch, input_dim, sequence_length)
        # -> (batch, sequence_length, input_dim)
        sequence = contaminated_signal.transpose(1, 2)

        bigru_sequence, _ = self.bigru(sequence)
        msca_features = self.msca(bigru_sequence)

        # Retains the extra ReLU from the original code.
        channel_first_features = msca_features.transpose(1, 2)
        activated_features = self.output_relu(channel_first_features)

        denoised_signal = self.decoder(activated_features)
        return denoised_signal


class Denoise(nn.Module):
    """Compatibility wrapper retaining the original external interface."""

    def __init__(
        self,
        input_dim_time,
        hidden_dim_time,
        channels_time,
        seq_length_time,
    ):
        super().__init__()

        self.bg_msca = BGMSCA(
            input_dim=input_dim_time,
            hidden_dim=hidden_dim_time,
            channels=channels_time,
            seq_length=seq_length_time,
        )

    def forward(self, time_input):
        return self.bg_msca(time_input)



