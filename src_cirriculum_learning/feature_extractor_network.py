import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiInputFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        
        # Branch sizes
        component_size = observation_space["components"].shape[0] * 4
        graph_size = observation_space["magVector"].shape[0]

        # --- Component branch ---
        self.components_net = nn.Sequential(
            nn.Linear(component_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # --- Magnitude branch ---
        self.mag_net = nn.Sequential(
            nn.Linear(graph_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # --- Phase branch ---
        self.phase_net = nn.Sequential(
            nn.Linear(graph_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # --- Fusion network with residual connection ---
        self.fusion_net = nn.Sequential(
            nn.Linear(288, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fusion_residual = nn.Linear(288, 256)   # residual shortcut
        self.fusion_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)  # stabilizes learning
        )

        # Optional: small dropout
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, obs):
        # --- Branch outputs ---
        c = self.components_net(obs["components"].view(obs["components"].shape[0], -1))
        m = self.mag_net(obs["magVector"])
        p = self.phase_net(obs["phaseVector"])

        # --- Fusion ---
        combined = torch.cat([c, m, p], dim=1)
        fused = self.fusion_net(combined) + self.fusion_residual(combined)  # residual
        fused = self.fusion_out(fused)
        fused = self.dropout(fused)

        return fused
