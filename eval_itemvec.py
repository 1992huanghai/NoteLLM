"""
使用预计算的 item_vec.csv 向量进行检索评估。
用于对比"其他系统的向量" vs "原始 T5" vs "NoteLLM 训练后"的性能。

注意：只评估 notes_val.csv 中出现的 note_id，确保和其他方法公平对比。
"""
import argparse
import json
import os
import csv
from typing import List, Optional, Dict, Set
from collections import defaultdict

import numpy as np


def load_valid_note_ids(notes_path: str) -> Set[str]:
	"""
	从 notes_val.csv 中加载有效的 note_id 列表。
	只评估这些 ID，确保和 Baseline/Trained 评估保持一致。
	"""
	print(f"Loading valid note IDs from {notes_path}...")
	valid_ids = set()
	
	if not os.path.exists(notes_path):
		print(f"Warning: {notes_path} not found, will use all IDs from item_vec.csv")
		return valid_ids
	
	with open(notes_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		try:
			first = next(reader)
		except StopIteration:
			return valid_ids
		
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
			
			for row in reader:
				if not row:
					continue
				note_id = row[i_id] if i_id is not None and i_id < len(row) else None
				if note_id and str(note_id).strip():
					valid_ids.add(str(note_id).strip())
		else:
			# 无表头：第一列是 note_id
			if first and len(first) > 0 and str(first[0]).strip():
				valid_ids.add(str(first[0]).strip())
			
			for row in reader:
				if row and len(row) > 0 and str(row[0]).strip():
					valid_ids.add(str(row[0]).strip())
	
	print(f"Loaded {len(valid_ids)} valid note IDs from {notes_path}")
	return valid_ids


def load_item_vectors(path: str, valid_ids: Optional[Set[str]] = None) -> tuple:
	"""
	加载 item_vec.csv。
	格式：note_id, vec1_vec2_vec3_..._vecN
	
	参数：
	  valid_ids: 如果提供，只加载这些 ID 的向量（用于和 notes_val.csv 对齐）
	
	返回：(note_ids: List[str], vectors: np.ndarray)
	"""
	print(f"Loading item vectors from {path}...")
	note_ids = []
	vectors = []
	skipped = 0
	
	with open(path, "r", encoding="utf-8") as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if not row or len(row) < 2:
				continue
			
			note_id = str(row[0]).strip()
			
			# 如果提供了 valid_ids，只加载在 valid_ids 中的 note_id
			if valid_ids is not None and note_id not in valid_ids:
				skipped += 1
				continue
			
			vec_str = row[1].strip()
			
			# 解析向量：vec1_vec2_vec3_... -> [vec1, vec2, vec3, ...]
			try:
				vec = np.array([float(x) for x in vec_str.split("_")], dtype=np.float32)
				note_ids.append(note_id)
				vectors.append(vec)
			except ValueError as e:
				if i < 10:  # 只打印前几个错误，避免刷屏
					print(f"Warning: Failed to parse row {i}: {e}")
				continue
			
			if (i + 1) % 50000 == 0:
				print(f"  Processed {i+1} rows, loaded {len(note_ids)} vectors...")
	
	if len(vectors) == 0:
		raise ValueError(f"No valid vectors loaded from {path}. Check file format.")
	
	vectors = np.vstack(vectors)
	print(f"Loaded {len(note_ids)} item vectors (skipped {skipped}), dim={vectors.shape[1]}")
	
	if valid_ids is not None:
		coverage = len(note_ids) / len(valid_ids) * 100 if valid_ids else 0
		print(f"Coverage: {len(note_ids)}/{len(valid_ids)} ({coverage:.2f}%) of valid IDs found in item_vec.csv")
	
	return note_ids, vectors


def load_related(path: str) -> Dict[str, List[str]]:
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
			has_header = any(h in {"note_id", "anchor", "a", "left", "source"} for h in header) and any(
				h in {"pos_note_id", "positive", "b", "right", "target"} for h in header
			)
			
			if has_header:
				def _idx(keys):
					for k in keys:
						if k in header:
							return header.index(k)
					return None
				
				i_a = _idx(["note_id", "anchor", "a", "left", "source", "note_a"])
				i_b = _idx(["pos_note_id", "positive", "b", "right", "target", "note_b"])
				
				for row in reader:
					if not row or len(row) < 2:
						continue
					anchor = row[i_a] if i_a is not None and i_a < len(row) else None
					positive = row[i_b] if i_b is not None and i_b < len(row) else None
					if anchor and positive:
						related_dict[str(anchor)].append(str(positive))
			else:
				# 无表头：按列序 [anchor, positive]
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


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
	"""L2 归一化向量（用于内积检索）。"""
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	norms = np.maximum(norms, 1e-9)  # 避免除零
	return vectors / norms


def eval_retrieval(
	note_ids: List[str],
	vectors: np.ndarray,
	related: Dict[str, List[str]],
	k: int = 10,
	dump_path: Optional[str] = None,
	normalize: bool = True,
) -> dict:
	"""
	用 item_vec.csv 的向量做检索评估。
	"""
	# L2 归一化（可选）
	if normalize:
		print("L2 normalizing vectors...")
		vectors = normalize_vectors(vectors)
	
	# 构建 id -> index 映射
	id_to_index = {note_id: i for i, note_id in enumerate(note_ids)}
	
	# FAISS 检索
	try:
		import faiss
		dim = vectors.shape[1]
		print(f"Building FAISS index (dim={dim})...")
		index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
		
		# Try GPU
		try:
			import torch
			if torch.cuda.is_available():
				print("Moving FAISS index to GPU...")
				res = faiss.StandardGpuResources()
				index = faiss.index_cpu_to_gpu(res, 0, index)
		except Exception:
			print("FAISS GPU unavailable, using CPU")
		
		index.add(vectors)
		topn = min(k + 1, vectors.shape[0])
		
		print(f"Searching top-{k} for {len(note_ids)} queries...")
		D, I = index.search(vectors, topn)
		
		hit_hits = 0
		recall_sum = 0.0
		count = 0
		rows = []
		
		for i, anchor_id in enumerate(note_ids):
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
			
			recommended = [note_ids[idx] for idx in rec_indices]
			
			if dump_path:
				rows.append((anchor_id, ",".join(recommended)))
			
			gt = related.get(anchor_id, [])
			if len(gt) > 0:
				gt_set = {str(g) for g in gt}
				rec_set = set(recommended)
				inter = len(gt_set & rec_set)
				hit_hits += int(inter > 0)
				recall_sum += (inter / max(1, len(gt_set)))
				count += 1
			
			if (i + 1) % 10000 == 0:
				print(f"  Processed {i+1}/{len(note_ids)} queries")
		
	except Exception as e:
		print(f"Error during retrieval: {e}")
		return {
			"recall_at_k": 0.0,
			"hit_rate_at_k": 0.0,
			"queries_with_gt": 0,
			"total_queries": len(note_ids),
		}
	
	# Write dump
	if dump_path and len(rows) > 0:
		dirname = os.path.dirname(dump_path)
		if dirname:
			os.makedirs(dirname, exist_ok=True)
		with open(dump_path, "w", encoding="utf-8") as f:
			for anchor_id, recs_str in rows:
				f.write(f"{anchor_id} {recs_str}\n")
		print(f"Recommendations saved to {dump_path}")
	
	den = max(1, count)
	hit_rate_at_k = hit_hits / den
	recall_at_k = recall_sum / den
	
	return {
		"recall_at_k": recall_at_k,
		"hit_rate_at_k": hit_rate_at_k,
		"queries_with_gt": count,
		"total_queries": len(note_ids),
	}


def main():
	parser = argparse.ArgumentParser(
		description="使用 item_vec.csv 预计算向量评估检索性能（只评估 notes_val.csv 中的 ID）"
	)
	parser.add_argument("--item_vec_path", required=True, help="item_vec.csv 路径")
	parser.add_argument("--notes_path", required=True, help="notes_val.csv 路径（用于确定评估哪些 ID）")
	parser.add_argument("--related_path", required=True, help="Ground-truth related notes 路径（JSON 或 CSV）")
	parser.add_argument("--k", type=int, default=5, help="Recall@K 的 K（默认 5）")
	parser.add_argument("--normalize", action="store_true", help="是否 L2 归一化向量（推荐开启）")
	parser.add_argument("--dump_recs_path", default=None, help="推荐结果保存路径（可选）")
	parser.add_argument("--max_items", type=int, default=0, help="最多评估多少条 items（0=全量，从 notes_val 中取前 N 个）")
	args = parser.parse_args()
	
	# Load valid note IDs from notes_val.csv
	valid_ids = load_valid_note_ids(args.notes_path)
	
	if args.max_items > 0 and args.max_items < len(valid_ids):
		print(f"Limiting to first {args.max_items} IDs from notes_val.csv")
		valid_ids = set(list(valid_ids)[:args.max_items])
	
	# Load item vectors (only for valid IDs)
	note_ids, vectors = load_item_vectors(args.item_vec_path, valid_ids=valid_ids)
	
	# Load ground truth
	related = load_related(args.related_path)
	
	# Evaluate
	results = eval_retrieval(
		note_ids=note_ids,
		vectors=vectors,
		related=related,
		k=args.k,
		dump_path=args.dump_recs_path,
		normalize=args.normalize,
	)
	
	# Print results
	print("\n" + "="*70)
	print(f"[Item Vector Evaluation Results] (from {args.item_vec_path})")
	print("="*70)
	print(f"Recall@{args.k}: {results['recall_at_k']:.4f}")
	print(f"Hit Rate@{args.k}: {results['hit_rate_at_k']:.4f}")
	print(f"Queries with GT: {results['queries_with_gt']}")
	print(f"Total queries: {results['total_queries']}")
	print(f"Coverage: {results['queries_with_gt']/max(1, results['total_queries'])*100:.2f}%")
	print("="*70)
	
	# Save JSON
	output_json = args.dump_recs_path.replace(".txt", "_itemvec_metrics.json") if args.dump_recs_path else "itemvec_metrics.json"
	results["source"] = args.item_vec_path
	results["k"] = args.k
	results["normalized"] = args.normalize
	
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)
	print(f"\n[Item Vec] Metrics saved to {output_json}")


if __name__ == "__main__":
	main()

