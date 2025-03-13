import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import numpy as np

class TemporalFusionTransformer(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
        loss: str = "RMSE",
        learning_rate: float = 0.03,
        sentiment_dim: int = 768,  # BERT embedding dimension
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Embedding layers
        self.categorical_embedding = nn.ModuleDict()
        self.continuous_embedding = nn.ModuleDict()
        
        # Sentiment embedding layer
        self.sentiment_embedding = nn.Linear(sentiment_dim, hidden_size)
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=attention_head_size,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # Output layers
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Loss function
        self.loss_fn = nn.MSELoss() if loss == "RMSE" else nn.L1Loss()
        
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        sentiment: torch.Tensor,
    ) -> torch.Tensor:
        # Process categorical features
        categorical_embeddings = []
        for name, embedding in self.categorical_embedding.items():
            categorical_embeddings.append(embedding(x[name]))
        
        # Process continuous features
        continuous_embeddings = []
        for name, embedding in self.continuous_embedding.items():
            continuous_embeddings.append(embedding(x[name]))
        
        # Process sentiment
        sentiment_embedding = self.sentiment_embedding(sentiment)
        
        # Combine all embeddings
        combined_embeddings = torch.cat(
            categorical_embeddings + continuous_embeddings + [sentiment_embedding],
            dim=-1
        )
        
        # Pass through transformer
        transformer_output = self.transformer(combined_embeddings)
        
        # Get predictions for the last timestep only
        predictions = self.output_layer(transformer_output[:, -1, :])
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        x, sentiment, y = batch
        y_hat = self(x, sentiment)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, sentiment, y = batch
        y_hat = self(x, sentiment)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def predict_step(self, batch, batch_idx):
        x, sentiment, _ = batch
        return self(x, sentiment) 
