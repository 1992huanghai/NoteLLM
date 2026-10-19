from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast


SPECIAL_COMPRESS_TOKEN = "<NOTE>"


@dataclass
class NoteBatch:
	note_ids: List[str]
	inputs: Dict[str, torch.Tensor]
	target_hashtags: Optional[Dict[str, torch.Tensor]] = None


class NoteLLM(nn.Module):
	"""Unified T5 with a special compression token for retrieval and generation."""
	def __init__(self, model_name: str = "t5-small", encode_max_len: int = 128, gen_max_len: int = 32):
		super().__init__()
		self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
		# add special compression token
		if SPECIAL_COMPRESS_TOKEN not in self.tokenizer.get_vocab():
			self.tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_COMPRESS_TOKEN]})
		self.model = T5ForConditionalGeneration.from_pretrained(model_name)
		self.model.resize_token_embeddings(len(self.tokenizer))
		self.encode_max_len = encode_max_len
		self.gen_max_len = gen_max_len

	def encode_notes(self, texts: List[str], device: torch.device) -> torch.Tensor:
		"""Encode notes and return embeddings from the final hidden state of the special token.
		If token not present, we prepend it to the input.
		"""
		prepended = [f"{SPECIAL_COMPRESS_TOKEN} {t}" if not t.strip().startswith(SPECIAL_COMPRESS_TOKEN) else t for t in texts]
		inputs = self.tokenizer(
			prepended,
			padding=True,
			truncation=True,
			max_length=self.encode_max_len,
			return_tensors="pt",
		)
		inputs = {k: v.to(device) for k, v in inputs.items()}
		outputs = self.model.encoder(**inputs)
		# get embedding at the first token position (SPECIAL_COMPRESS_TOKEN)
		emb = outputs.last_hidden_state[:, 0, :]
		return nn.functional.normalize(emb, p=2, dim=-1)

	def forward_hashtag_loss(self, texts: List[str], hashtags: List[List[str]], device: torch.device) -> torch.Tensor:
		inputs = self.tokenizer(
			[f"summarize and generate hashtags: {t}" for t in texts],
			padding=True,
			truncation=True,
			max_length=self.encode_max_len,
			return_tensors="pt",
		)
		labels = self.tokenizer(
			[" ".join(hs) for hs in hashtags],
			padding=True,
			truncation=True,
			max_length=self.gen_max_len,
			return_tensors="pt",
		)["input_ids"]
		# ensure correct dtype
		labels = labels.long()
		# mask pad tokens to ignore index
		pad_id = self.tokenizer.pad_token_id
		if pad_id is not None:
			labels[labels == pad_id] = -100
		# safety: clamp any accidental out-of-range labels
		vocab_size = self.model.get_output_embeddings().weight.size(0)
		labels = torch.where((labels >= vocab_size) & (labels != -100), torch.full_like(labels, self.tokenizer.unk_token_id), labels)

		inputs = {k: v.to(device) for k, v in inputs.items()}
		labels = labels.to(device)
		outputs = self.model(**inputs, labels=labels)
		return outputs.loss

	def contrastive_loss(self, emb_anchor: torch.Tensor, emb_positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
		"""InfoNCE with in-batch negatives. Make sure sizes match and are > 0."""
		n = min(emb_anchor.size(0), emb_positive.size(0))
		if n == 0:
			return emb_anchor.new_tensor(0.0)
		emb_anchor = nn.functional.normalize(emb_anchor[:n], p=2, dim=-1)
		emb_positive = nn.functional.normalize(emb_positive[:n], p=2, dim=-1)
		logits = emb_anchor @ emb_positive.t() / temperature
		target = torch.arange(n, device=logits.device)
		return nn.functional.cross_entropy(logits, target)
