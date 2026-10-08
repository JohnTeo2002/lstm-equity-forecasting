try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


if TORCH_AVAILABLE:
    class EquityLSTM(nn.Module):
        """
        Stacked LSTM neural network architecture for multi-step equity forecasting.
        """

        def __init__(
            self,
            input_dim: int = 5,
            hidden_dim: int = 128,
            num_layers: int = 2,
            output_dim: int = 5,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.output_dim = output_dim

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            self.fc_layers = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            x shape: (batch_size, sequence_length, input_dim)
            Returns shape: (batch_size, output_dim)
            """
            lstm_out, _ = self.lstm(x)
            last_hidden_state = lstm_out[:, -1, :]
            out = self.fc_layers(last_hidden_state)
            return out
else:
    class EquityLSTM:
        """Stub class when PyTorch is not yet installed."""
        def __init__(self, *args, **kwargs):
            pass

