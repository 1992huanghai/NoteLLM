"""
Baseline evaluation with different pooling strategies.
对比不同 pooling 方法对原始 T5 检索性能的影响。
"""
import argparse
import json
import os
import csv
from typing import List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from transformers import T5ForConditionalGeneration, T5TokenizerFast


class T5BaselineEncoder:
	"""原始 T5 Encoder，支持多种 pooling 策略。"""
	
	def __init__(self, model_name: str = "t5-small", encode_max_len: int = 128, pooling: str = "first"):
		self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
		self.model = T5ForConditionalGeneration.from_pretrained(model_name)
		self.encode_max_len = encode_max_len
		self.pooling = pooling  # "first", "mean", "max", "last"
		
		# 如果用 "first" pooling，添加特殊 token <NOTE>
		if pooling == "first":
			special_token = "<NOTE>"
			if special_token not in self.tokenizer.get_vocab():
				self.tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
				self.model.resize_token_embeddings(len(self.tokenizer))
	
	def encode_notes(self, texts: List[str], device: torch.device) -> torch.Tensor:
		"""用不同 pooling 策略提取 embedding。"""
		# Prepend special token if using "first" pooling
		if self.pooling == "first":
			texts = [f"<NOTE> {t}" for t in texts]
		
		# Tokenize
		inputs = self.tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=self.encode_max_len,
			return_tensors="pt",
		)
		inputs = {k: v.to(device) for k, v in inputs.items()}
		
		# Forward through encoder
		outputs = self.model.encoder(**inputs)
		hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
		
		# Apply pooling
		if self.pooling == "first":
			# Use first token (<NOTE>)
			emb = hidden_states[:, 0, :]
		
		elif self.pooling == "mean":
			# Mean pooling (exclude padding)
			attention_mask = inputs["attention_mask"]  # [batch, seq_len]
			mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
			sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
			sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
			emb = sum_hidden / sum_mask
		
		elif self.pooling == "max":
			# Max pooling (exclude padding)
			attention_mask = inputs["attention_mask"]
			mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
			hidden_states = hidden_states.clone()
			hidden_states[mask_expanded == 0] = -1e9  # Set padding to very negative
			emb = torch.max(hidden_states, dim=1)[0]
		
		elif self.pooling == "last":
			# Use last non-padding token
			attention_mask = inputs["attention_mask"]
			seq_lengths = attention_mask.sum(dim=1) - 1  # Last token index
			batch_size = hidden_states.size(0)
			emb = hidden_states[torch.arange(batch_size), seq_lengths]
		
		else:
			raise ValueError(f"Unknown pooling: {self.pooling}")
		
		# L2 normalize
		return nn.functional.normalize(emb, p=2, dim=-1)


def load_notes(path: str):
	"""加载 notes（兼容 CSV 和 JSONL）。"""
	items = []
	if path.endswith(".csv"):
		with open(path, "r", encoding="utf-8", newline="") as f:
			reader = csv.reader(f)
			try:
				first = next(reader)
			except StopIteration:
				return []
			
			def _norm(s: str) -> str:
				return (s or "").strip().lower()
			
			header = [_norm(x) for x in first]
			has_header = any(h in {"note_id", "id"} for h in header)
			
			if has_header:
				def _idx(keys):
					for k in keys:
						if k in header:
							return header.index(k)
					return None
				
				i_id = _idx(["note_id", "id"])
				i_content = _idx(["content", "text"])
				
				for row in reader:
					if not row:
						continue
					note_id = row[i_id] if i_id is not None and i_id < len(row) else None
					content = row[i_content] if i_content is not None and i_content < len(row) else ""
					items.append({"note_id": note_id, "content": content or ""})
			else:
				def _emit(row):
					if not row or len(row) < 2:
						return
					items.append({"note_id": str(row[0]), "content": row[1] or ""})
				
				_emit(first)
				for row in reader:
					_emit(row)
	else:
		with open(path, "r") as f:
			for line in f:
				items.append(json.loads(line))
	
	print(f"Loaded {len(items)} notes from {path}")
	return items


def load_related(path: str):
	"""加载 ground-truth（支持 JSON 和 CSV）。"""
	if path.endswith(".json"):
		with open(path, "r") as f:
			return json.load(f)
	elif path.endswith(".csv"):
		related_dict = defaultdict(list)
		with open(path, "r", newline="") as f:
			reader = csv.reader(f)
			first = next(reader, None)
			if not first:
				return {}
			
			has_header = any(h.strip().lower() in {"note_id", "anchor"} for h in first)
			if not has_header:
				if len(first) >= 2:
					related_dict[str(first[0])].append(str(first[1]))
			
			for row in reader:
				if len(row) >= 2:
					related_dict[str(row[0])].append(str(row[1]))
		
		return dict(related_dict)
	else:
		raise ValueError(f"Unsupported format: {path}")


def note_text(n: dict) -> str:
	return n.get("content") or n.get("text") or ""


def eval_retrieval(
	encoder: T5BaselineEncoder,
	notes: List[dict],
	related: dict,
	k: int,
	batch_size: int,
	device: torch.device,
) -> dict:
	"""评估检索性能。"""
	texts = [note_text(n) for n in notes]
	emb_chunks = []
	
	encoder.model.eval()
	print(f"[Pooling={encoder.pooling}] Encoding {len(texts)} notes...")
	
	with torch.no_grad():
		for i in range(0, len(texts), batch_size):
			chunk = texts[i:i + batch_size]
			chunk_emb = encoder.encode_notes(chunk, device)
			emb_chunks.append(chunk_emb.cpu().numpy())
			if (i // batch_size) % 50 == 0:
				print(f"  {i}/{len(texts)}")
	
	emb = np.vstack(emb_chunks).astype(np.float32)
	print(f"Embedding shape: {emb.shape}")
	
	# FAISS retrieval
	try:
		import faiss
		dim = emb.shape[1]
		index = faiss.IndexFlatIP(dim)
		
		if torch.cuda.is_available():
			try:
				res = faiss.StandardGpuResources()
				index = faiss.index_cpu_to_gpu(res, 0, index)
			except:
				pass
		
		index.add(emb)
		D, I = index.search(emb, k + 1)
		
		id_to_index = {str(n.get("note_id", n.get("id"))): i for i, n in enumerate(notes)}
		hit_hits = 0
		recall_sum = 0.0
		count = 0
		
		for i, n in enumerate(notes):
			anchor_id = str(n.get("note_id", n.get("id")))
			anchor_idx = id_to_index.get(anchor_id, -1)
			
			rec_indices = [int(j) for j in I[i] if int(j) != anchor_idx][:k]
			recommended = [str(notes[idx].get("note_id", notes[idx].get("id"))) for idx in rec_indices]
			
			gt = related.get(anchor_id, [])
			if len(gt) > 0:
				gt_set = {str(g) for g in gt}
				rec_set = set(recommended)
				inter = len(gt_set & rec_set)
				hit_hits += int(inter > 0)
				recall_sum += inter / len(gt_set)
				count += 1
		
		return {
			"recall_at_k": recall_sum / max(1, count),
			"hit_rate_at_k": hit_hits / max(1, count),
			"queries_with_gt": count,
		}
	
	except Exception as e:
		print(f"Error: {e}")
		return {"recall_at_k": 0.0, "hit_rate_at_k": 0.0, "queries_with_gt": 0}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--notes_path", required=True)
	parser.add_argument("--related_path", required=True)
	parser.add_argument("--model_name_or_path", default="t5-small")
	parser.add_argument("--pooling", default="first", choices=["first", "mean", "max", "last"],
	                    help="Pooling strategy: first (<NOTE> token), mean, max, last")
	parser.add_argument("--k", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--encode_max_len", type=int, default=128)
	parser.add_argument("--max_notes", type=int, default=0)
	args = parser.parse_args()
	
	notes = load_notes(args.notes_path)
	if args.max_notes > 0:
		notes = notes[:args.max_notes]
	
	related = load_related(args.related_path)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	encoder = T5BaselineEncoder(args.model_name_or_path, args.encode_max_len, args.pooling)
	encoder.model.to(device)
	
	results = eval_retrieval(encoder, notes, related, args.k, args.batch_size, device)
	
	print("\n" + "="*60)
	print(f"[Baseline - {args.pooling.upper()} pooling]")
	print("="*60)
	print(f"Recall@{args.k}: {results['recall_at_k']:.4f}")
	print(f"Hit Rate@{args.k}: {results['hit_rate_at_k']:.4f}")
	print(f"Queries with GT: {results['queries_with_gt']}")
	print("="*60)
	
	# Save
	output_json = f"baseline_pooling_{args.pooling}_metrics.json"
	with open(output_json, "w") as f:
		json.dump({**results, "pooling": args.pooling, "k": args.k}, f, indent=2)
	print(f"\nSaved to {output_json}")


if __name__ == "__main__":
	main()

