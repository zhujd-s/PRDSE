from collections import deque

import numpy as np
import torch


class SearchContextBuffer():
	def __init__(self, seq_len, context_dim, dtype=torch.float32):
		self.seq_len = seq_len
		self.context_dim = context_dim
		self.dtype = dtype
		self.buffer = deque(maxlen=seq_len)

	def reset(self):
		self.buffer.clear()

	def append(self, context_vec):
		context_array = np.asarray(context_vec, dtype=np.float32).reshape(-1)
		if(context_array.shape[0] != self.context_dim):
			raise ValueError(f"context_dim mismatch: expect {self.context_dim}, got {context_array.shape[0]}")
		self.buffer.append(context_array)

	def get_sequence_array(self):
		seq_array = np.zeros((self.seq_len, self.context_dim), dtype=np.float32)
		start_idx = self.seq_len - len(self.buffer)
		for index, context_vec in enumerate(self.buffer):
			seq_array[start_idx + index] = context_vec
		return seq_array

	def get_sequence_tensor(self, device=None):
		return torch.tensor(self.get_sequence_array(), dtype=self.dtype, device=device)
