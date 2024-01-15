import torch
import torch.nn as nn


class InputProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        output = self.fc(x)
        
        return output


class OutputProjectionLayer(nn.Module):
    def __init__(self, input_dim, num_clip_tokens, prompt_embeddings_dim) -> None:
        super().__init__()
        self.num_clip_tokens = num_clip_tokens
        self.prompt_embeddings_dim = prompt_embeddings_dim

        hidden_dim = 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                  d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                  dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
        self.fc2 = nn.Linear(hidden_dim, prompt_embeddings_dim)
        self.query_embs = nn.Parameter(torch.randn(1, num_clip_tokens, hidden_dim))
        
    def forward(self, x):
        x = self.fc1(x.float())  # [b_s, 32, hidden_dim] = [b_s, 32, 512]
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))  # [b_s, num_clip_tokens, hidden_dim] = [b_s, 77, 512]
        output = self.fc2(x)  # [b_s, num_clip_tokens, prompt_embeddings_dim] = [b_s, 77, 768]

        return output
