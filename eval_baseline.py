"""
Baseline evaluation: 用原始 T5 预训练模型（未经 NoteLLM 训练）提取 embedding 做检索评估。
用于对比"对比学习 + CSFT 训练"相对于"纯预训练模型"的提升效果。
"""
import argparse
import json
import os
import csv
from typing import List, Optional
from collections import defaultdict

import torch
import numpy as np

from .modeling import NoteLLM


def load_notes(path: str):
	"""
	加载 notes（兼容有表头/无表头 CSV 和 JSONL）。
	"""
	items: List[dict] = []
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
			has_header = any(h in {"note_id", "id", "content", "text", "hashtags", "tags"} for h in header)
			
			if has_header:
				def _idx(keys):
					for k in keys:
						if k in header:
							return header.index(k)
					return None
				
				i_id = _idx(["note_id", "id"])
				i_content = _idx(["content", "text"])
				i_hashtags = _idx(["hashtags", "tags"])
				
				for row in reader:
					if not row:
						continue
					note_id = row[i_id] if i_id is not None and i_id < len(row) else None
					content = row[i_content] if i_content is not None and i_content < len(row) else ""
					hashtags = row[i_hashtags] if i_hashtags is not None and i_hashtags < len(row) else ""
					items.append({
						"note_id": note_id,
						"content": content or "",
						"hashtags": hashtags or "",
					})
			else:
				def _emit(row):
					if not row:
						return
					note_id = row[0] if len(row) > 0 else None
					if note_id is None or str(note_id).strip() == "":
						return
					content = row[1] if len(row) > 1 else ""
					hashtags = row[2] if len(row) > 2 else ""
					items.append({
						"note_id": str(note_id),
						"content": content or "",
						"hashtags": hashtags or "",
					})
				
				_emit(first)
				for row in reader:
					_emit(row)
	else:
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				items.append(json.loads(line))
	
	print(f"Loaded {len(items)} notes from {path}")
	return items


def load_related(path: str):
	"""
	加载 ground-truth related notes（支持 JSON 和 CSV）。
	"""
	if path.endswith(".json"):
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	elif path.endswith(".csv"):
		related_dict = defaultdict(list)
		with open(path, "r", encoding="utf-8", newline="") as f:
			reader = csv.reader(f)
			try:
				first = next(reader)
			except StopIteration:
				return {}
			
			def _norm(s: str) -> str:
				return (s or "").strip().lower()
			
			header = [_norm(x) for x in first]
			has_header = any(h in {"note_id", "anchor", "a"} for h in header) and any(
				h in {"pos_note_id", "positive", "b"} for h in header
			)
			
			if has_header:
				def _idx(keys):
					for k in keys:
						if k in header:
							return header.index(k)
					return None
				
				i_a = _idx(["note_id", "anchor", "a"])
				i_b = _idx(["pos_note_id", "positive", "b"])
				
				for row in reader:
					if not row or len(row) < 2:
						continue
					anchor = row[i_a] if i_a is not None and i_a < len(row) else None
					positive = row[i_b] if i_b is not None and i_b < len(row) else None
					if anchor and positive:
						related_dict[str(anchor)].append(str(positive))
			else:
				def _emit(row):
					if not row or len(row) < 2:
						return
					anchor = row[0]
					positive = row[1]
					if anchor and positive:
						related_dict[str(anchor)].append(str(positive))
				
				_emit(first)
				for row in reader:
					_emit(row)
		
		return dict(related_dict)
	else:
		raise ValueError(f"Unsupported format: {path}")


def note_text(n: dict) -> str:
	text = n.get("content") or n.get("text")
	if not text:
		title = n.get("title", "")
		content = n.get("content", "")
		combo = (title + ". " + content).strip()
		text = combo if combo else ""
	return text


def eval_baseline_retrieval(
	model: NoteLLM,
	notes: List[dict],
	related: dict,
	k: int = 10,
	batch_size: int = 32,
	dump_path: Optional[str] = None,
) -> dict:
	"""
	用原始 T5 模型（只加载预训练权重，未经 NoteLLM 训练）提取 embedding 并做检索评估。
	"""
	texts = [note_text(n) for n in notes]
	emb_chunks = []
	
	model.model.eval()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	
	print(f"[Baseline] Encoding {len(texts)} notes using pretrained T5 encoder...")
	with torch.no_grad():
		for i in range(0, len(texts), batch_size):
			chunk = texts[i:i + batch_size]
			chunk_emb = model.encode_notes(chunk, device)
			emb_chunks.append(chunk_emb.detach().cpu().numpy())
			del chunk_emb
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			if (i // batch_size) % 50 == 0:
				print(f"  Encoded {i}/{len(texts)} notes")
	
	emb = np.vstack(emb_chunks) if len(emb_chunks) > 0 else np.zeros((0, 0), dtype=np.float32)
	print(f"[Baseline] Embedding shape: {emb.shape}")
	
	# Map note_id to index
	id_to_index = {str(n.get("note_id", n.get("id"))): i for i, n in enumerate(notes)}
	
	# Retrieval
	hit_hits = 0
	recall_sum = 0.0
	count = 0
	rows = []
	
	emb = emb.astype(np.float32, copy=False)
	
	# Try FAISS
	use_faiss = False
	try:
		import faiss
		use_faiss = emb.shape[0] > 0 and emb.shape[1] > 0
	except Exception:
		use_faiss = False
	
	if use_faiss:
		dim = int(emb.shape[1])
		print(f"[Baseline] Using FAISS (dim={dim})")
		index = faiss.IndexFlatIP(dim)
		
		if torch.cuda.is_available():
			try:
				print('[Baseline] Moving FAISS index to GPU...')
				res = faiss.StandardGpuResources()
				index = faiss.index_cpu_to_gpu(res, 0, index)
			except Exception:
				print('[Baseline] FAISS GPU failed, using CPU')
		
		index.add(emb)
		topn = min(k + 1, emb.shape[0])
		
		print(f"[Baseline] Searching top-{k} for {len(notes)} queries...")
		D, I = index.search(emb, topn)
		
		for i, n in enumerate(notes):
			anchor_id = str(n.get("note_id", n.get("id")))
			anchor_idx = id_to_index.get(anchor_id, -1)
			
			# Exclude self
			rec_indices = []
			for j in I[i]:
				jj = int(j)
				if jj == anchor_idx:
					continue
				rec_indices.append(jj)
				if len(rec_indices) >= k:
					break
			
			recommended = [str(notes[idx].get("note_id", notes[idx].get("id"))) for idx in rec_indices]
			
			if dump_path:
				rows.append((anchor_id, ",".join(recommended)))
			
			gt = related.get(str(anchor_id), [])
			if len(gt) > 0:
				gt_set = {str(g) for g in gt}
				rec_set = set(recommended)
				inter = len(gt_set & rec_set)
				hit_hits += int(inter > 0)
				recall_sum += (inter / max(1, len(gt_set)))
				count += 1
			
			if (i + 1) % 5000 == 0:
				print(f"  Processed {i+1}/{len(notes)} queries")
	else:
		print('[Baseline] Using numpy fallback (FAISS unavailable)')
		for i, n in enumerate(notes):
			anchor_id = str(n.get("note_id", n.get("id")))
			scores = emb @ emb[i]
			anchor_idx = id_to_index.get(anchor_id, -1)
			
			if k < len(scores):
				topk_idx = np.argpartition(-scores, kth=k)[:k+1]
				order = topk_idx[np.argsort(-scores[topk_idx])]
			else:
				order = np.argsort(-scores)
			
			recommended = [str(notes[idx].get("note_id", notes[idx].get("id"))) for idx in order if idx != anchor_idx][:k]
			
			if dump_path:
				rows.append((anchor_id, ",".join(recommended)))
			
			gt = related.get(str(anchor_id), [])
			if len(gt) > 0:
				gt_set = {str(g) for g in gt}
				rec_set = set(recommended)
				inter = len(gt_set & rec_set)
				hit_hits += int(inter > 0)
				recall_sum += (inter / max(1, len(gt_set)))
				count += 1
	
	# Write dump
	if dump_path and len(rows) > 0:
		dirname = os.path.dirname(dump_path)
		if dirname:
			os.makedirs(dirname, exist_ok=True)
		with open(dump_path, "w", encoding="utf-8") as f:
			for anchor_id, recs_str in rows:
				f.write(f"{anchor_id} {recs_str}\n")
		print(f"[Baseline] Recommendations saved to {dump_path}")
	
	den = max(1, count)
	hit_rate_at_k = hit_hits / den
	recall_at_k = recall_sum / den
	
	return {
		"hit_rate_at_k": hit_rate_at_k,
		"recall_at_k": recall_at_k,
		"queries_with_gt": count,
		"total_queries": len(notes),
	}


def main():
	parser = argparse.ArgumentParser(
		description="Baseline evaluation: 用原始 T5 预训练模型（未经 NoteLLM 训练）评估检索性能"
	)
	parser.add_argument("--notes_path", required=True, help="验证集 notes 路径（CSV 或 JSONL）")
	parser.add_argument("--related_path", required=True, help="Ground-truth related notes 路径（JSON 或 CSV）")
	parser.add_argument("--model_name_or_path", default="t5-small", help="预训练模型名称（默认 t5-small）")
	parser.add_argument("--k", type=int, default=5, help="Recall@K 的 K（默认 5）")
	parser.add_argument("--batch_size", type=int, default=32, help="编码 batch size（默认 32）")
	parser.add_argument("--encode_max_len", type=int, default=128, help="输入 max length（默认 128）")
	parser.add_argument("--max_notes", type=int, default=None, help="最多评估多少条 notes（用于快速测试，默认全量）")
	parser.add_argument("--dump_recs_path", default=None, help="推荐结果保存路径（可选）")
	args = parser.parse_args()
	
	# Load data
	notes = load_notes(args.notes_path)
	if args.max_notes and args.max_notes > 0:
		notes = notes[:args.max_notes]
		print(f"[Baseline] Limited to first {len(notes)} notes for testing")
	
	related = load_related(args.related_path)
	
	# Load pretrained T5 (without NoteLLM training)
	print(f"[Baseline] Loading pretrained model: {args.model_name_or_path}")
	print("[Baseline] NOTE: 这是未经 NoteLLM 训练的原始预训练权重！")
	model = NoteLLM(
		model_name=args.model_name_or_path,
		encode_max_len=args.encode_max_len,
		gen_max_len=32,  # 不需要生成，但保持接口一致
	)
	
	# Evaluate
	results = eval_baseline_retrieval(
		model=model,
		notes=notes,
		related=related,
		k=args.k,
		batch_size=args.batch_size,
		dump_path=args.dump_recs_path,
	)
	
	# Print results
	print("\n" + "="*60)
	print(f"[Baseline Evaluation Results] (Pretrained {args.model_name_or_path})")
	print("="*60)
	print(f"Recall@{args.k}: {results['recall_at_k']:.4f}")
	print(f"Hit Rate@{args.k}: {results['hit_rate_at_k']:.4f}")
	print(f"Queries with GT: {results['queries_with_gt']}")
	print(f"Total queries: {results['total_queries']}")
	print("="*60)
	
	# Save JSON
	output_json = args.dump_recs_path.replace(".txt", "_baseline_metrics.json") if args.dump_recs_path else "baseline_metrics.json"
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)
	print(f"\n[Baseline] Metrics saved to {output_json}")


if __name__ == "__main__":
	main()

