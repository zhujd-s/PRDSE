import math

import torch
import torch.nn as nn


class TransformerMetricController(nn.Module):
	def __init__(self, context_dim, seq_len, num_metrics=5, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.0, default_metric_weights=None, default_lambda_ext=0.1):
		super().__init__()
		self.context_dim = context_dim
		self.seq_len = seq_len
		self.num_metrics = num_metrics
		self.d_model = d_model
		self.default_lambda_ext = default_lambda_ext
		if(default_metric_weights is None):
			default_metric_weights = torch.full((num_metrics,), 1.0 / num_metrics, dtype=torch.float32)
		self.default_metric_weights = default_metric_weights.float()

		self.input_proj = nn.Linear(context_dim, d_model)
		self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.norm = nn.LayerNorm(d_model)
		self.metric_head = nn.Linear(d_model, num_metrics)
		self.lambda_head = nn.Linear(d_model, 1)

		self._reset_parameters()

	def _reset_parameters(self):
		nn.init.xavier_uniform_(self.input_proj.weight)
		nn.init.zeros_(self.input_proj.bias)
		nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
		nn.init.xavier_uniform_(self.metric_head.weight, gain=0.05)
		nn.init.zeros_(self.metric_head.bias)
		nn.init.xavier_uniform_(self.lambda_head.weight, gain=0.05)
		nn.init.zeros_(self.lambda_head.bias)

		with torch.no_grad():
			self.metric_head.bias.copy_(torch.log(self.default_metric_weights))
			self.lambda_head.bias.fill_(torch.logit(torch.tensor(self.default_lambda_ext, dtype=torch.float32), eps=1e-6).item())

	def forward(self, context_seq):
		if(context_seq.dim() == 2):
			context_seq = context_seq.unsqueeze(0)

		if(context_seq.shape[-2] != self.seq_len or context_seq.shape[-1] != self.context_dim):
			raise ValueError(
				f"context_seq shape mismatch: expect (*, {self.seq_len}, {self.context_dim}), got {tuple(context_seq.shape)}"
			)

		encoded = self.input_proj(context_seq) * math.sqrt(self.d_model)
		encoded = encoded + self.pos_embedding[:, :encoded.shape[1], :]
		encoded = self.encoder(encoded)
		summary = self.norm(encoded[:, -1, :])

		metric_logits = self.metric_head(summary)
		metric_weights = torch.softmax(metric_logits, dim=-1)
		lambda_ext = torch.sigmoid(self.lambda_head(summary))
		return metric_weights.squeeze(0), lambda_ext.squeeze(0)
