import argparse
import json
import os
import csv
from typing import List, Optional

import torch

import numpy as np
from sklearn.metrics import f1_score

from .modeling import NoteLLM


def load_notes(path: str):
	"""
	加载 notes，用于评估。
	支持：
	- CSV（有表头）：列为 note_id, content, hashtags
	- CSV（无表头）：按列序 [note_id, content, hashtags]
	- JSONL：一行一个 JSON 对象（兼容旧格式）
	"""
	items: List[dict] = []
	if path.endswith(".csv"):
		# 兼容有表头/无表头的 notes CSV（避免 for-else 缩进误绑定导致只读到 1 行）
		with open(path, "r", encoding="utf-8", newline="") as f:
			reader = csv.reader(f)
			try:
				first = next(reader)
			except StopIteration:
				return []

			def _norm(s: str) -> str:
				return (s or "").strip().lower()

			header = [_norm(x) for x in first]
			has_header = any(h in {"note_id", "id"} for h in header) and any(h in {"content", "text"} for h in header)

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
					if note_id is None or str(note_id).strip() == "":
						continue
					content = row[i_content] if i_content is not None and i_content < len(row) else ""
					hashtags = row[i_hashtags] if i_hashtags is not None and i_hashtags < len(row) else ""
					items.append({"note_id": str(note_id), "content": content or "", "hashtags": hashtags or ""})
			else:
				def _emit(row):
					if not row:
						return
					note_id = row[0] if len(row) > 0 else None
					if note_id is None or str(note_id).strip() == "":
						return
					content = row[1] if len(row) > 1 else ""
					hashtags = row[2] if len(row) > 2 else ""
					items.append({"note_id": str(note_id), "content": content or "", "hashtags": hashtags or ""})

				_emit(first)
				for row in reader:
					_emit(row)
	else:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			items.append(json.loads(line))
	return items


def load_related(path: str):
	"""
	加载 ground-truth related notes。
	支持：
	- JSON：{note_id: [related_ids]}
	- CSV：列为 note_id, pos_note_id（或 anchor, positive 等别名）
	"""
	if path.endswith(".json"):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)
	elif path.endswith(".csv"):
		# 读取 CSV pairs 并转换成 {note_id: [related_ids]}
		from collections import defaultdict
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
			has_header = any(h in {"note_id", "anchor", "a", "left", "source", "note_a"} for h in header) and any(
				h in {"pos_note_id", "positive", "b", "right", "target", "note_b"} for h in header
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
				# 无表头，按列序 [anchor, positive]
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
		raise ValueError(f"Unsupported file format for related_path: {path} (expected .json or .csv)")


def note_text(n: dict) -> str:
	text = n.get("content") or n.get("text")
	if not text:
		title = n.get("title", "")
		content = n.get("content", "")
		combo = (title + ". " + content).strip()
		text = combo if combo else ""
	return text


def eval_retrieval(model: NoteLLM, notes: List[dict], related: dict, k: int = 10, batch_size: int = 32, dump_path: Optional[str] = None, search_batch_size: Optional[int] = None, notes_global: Optional[List[dict]] = None) -> float:
	# Use global notes for indexing if provided, otherwise use notes (validation set)
	index_notes = notes_global if notes_global is not None else notes
	query_notes = notes  # Always query with validation set
	
	# Encode all global notes for indexing
	texts_global = [note_text(n) for n in index_notes]
	emb_chunks = []
	model.model.eval()
	if torch.cuda.is_available():
		model.to(torch.device('cuda'))
	print(f'start encode global notes (total: {len(texts_global)})')
	with torch.no_grad():
		for i in range(0, len(texts_global), batch_size):
			chunk = texts_global[i:i + batch_size]
			chunk_emb = model.encode_notes(chunk, model.model.device)
			emb_chunks.append(chunk_emb.detach().cpu().numpy())
			# release GPU memory between batches
			del chunk_emb
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
	emb_global = np.vstack(emb_chunks) if len(emb_chunks) > 0 else np.zeros((0, 0), dtype=np.float32)
	
	# Encode validation notes for querying
	texts_query = [note_text(n) for n in query_notes]
	emb_chunks = []
	print(f'start encode query notes (total: {len(texts_query)})')
	with torch.no_grad():
		for i in range(0, len(texts_query), batch_size):
			chunk = texts_query[i:i + batch_size]
			chunk_emb = model.encode_notes(chunk, model.model.device)
			emb_chunks.append(chunk_emb.detach().cpu().numpy())
			# release GPU memory between batches
			del chunk_emb
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
	emb_query = np.vstack(emb_chunks) if len(emb_chunks) > 0 else np.zeros((0, 0), dtype=np.float32)

	# map ids to indices for global notes (for indexing) - normalize to str for robustness
	id_to_index_global = {str(n.get("note_id", n.get("id"))): i for i, n in enumerate(index_notes)}
	# Metrics:
	# - hit_rate@k: any(gt) in topk → 1 else 0, averaged over queries with non-empty gt
	# - recall@k: |gt ∩ topk| / |gt|, averaged over queries with non-empty gt
	hit_hits = 0
	recall_sum = 0.0
	count = 0
	rows = []
	print("start retrieval")
	# ensure float32 for faiss
	emb_global = emb_global.astype(np.float32, copy=False)
	emb_query = emb_query.astype(np.float32, copy=False)

	use_faiss = False
	try:
		import faiss  # type: ignore
		use_faiss = emb_global.shape[0] > 0 and emb_global.shape[1] > 0
	except Exception:
		use_faiss = False

	if use_faiss:
		dim = int(emb_global.shape[1])
		print(f'FAISS index dimension: {dim}')
		index = faiss.IndexFlatIP(dim)
		# move to GPU if possible
		if torch.cuda.is_available():
			try:
				print('move FAISS index to GPU')
				res = faiss.StandardGpuResources()
				index = faiss.index_cpu_to_gpu(res, 0, index)
			except Exception:
				pass
		print(f'Adding {emb_global.shape[0]} items to FAISS index')
		index.add(emb_global)
		is_gpu = hasattr(index, 'getDevice')
		if is_gpu:
			print('FAISS on GPU, device:', index.getDevice())
		else:
			print('FAISS on CPU')
		topn = min(k + 1, emb_global.shape[0])
		# batched search to reduce peak memory
		bs = search_batch_size if (search_batch_size is not None and search_batch_size > 0) else min(8192, emb_query.shape[0] if emb_query.shape[0] > 0 else 8192)
		print(f'faiss batched search start, batch_size = {bs}')
		print(f'Searching {emb_query.shape[0]} query notes against {emb_global.shape[0]} indexed notes')
		total = emb_query.shape[0]
		done = 0
		for start in range(0, total, bs):
			end = min(start + bs, total)
			# Search using query embeddings against global index
			D_chunk, I_chunk = index.search(emb_query[start:end], topn)
			# transform to note ids and compute recall
			print('transform to note ids and compute recall')
			for row_offset, n in enumerate(query_notes[start:end]):
				query_idx = start + row_offset
				anchor_id = str(n.get("note_id", n.get("id")))
				# exclude self from recommendations if query note is in global index
				anchor_global_idx = id_to_index_global.get(anchor_id, -1)
				rec_indices = []
				for j in I_chunk[row_offset]:
					jj = int(j)
					if jj == anchor_global_idx:
						continue
					rec_indices.append(jj)
					if len(rec_indices) >= k:
						break
				recommended = [str(index_notes[i].get("note_id", index_notes[i].get("id"))) for i in rec_indices]
				if dump_path:
					rows.append((anchor_id, ",".join(recommended)))
				gt = related.get(str(anchor_id), [])
				"""
				print ("=======")
				print(recommended)
				print(gt)
				print ("=======")
				"""
				if len(gt) > 0:
					gt_set = {str(g) for g in gt}
					rec_set = set(recommended)
					inter = len(gt_set & rec_set)
					hit_hits += int(inter > 0)
					recall_sum += (inter / max(1, len(gt_set)))
					count += 1
			done = end
			#if done % 10000 == 0 or done == total:
			print('faiss search progressed:', done, '/', total)
	else:
		# fallback to numpy similarity if faiss is unavailable
		print('Using numpy fallback (FAISS unavailable)')
		for n in query_notes:
			anchor_id = str(n.get("note_id", n.get("id")))
			query_idx = query_notes.index(n)
			# Search query embedding against global embeddings
			scores = emb_global @ emb_query[query_idx]
			anchor_global_idx = id_to_index_global.get(anchor_id, -1)
			if k < len(scores):
				topk_idx = np.argpartition(-scores, kth=k)[:k+1]
				order = topk_idx[np.argsort(-scores[topk_idx])]
			else:
				order = np.argsort(-scores)
			recommended = [str(index_notes[i].get("note_id", index_notes[i].get("id"))) for i in order if i != anchor_global_idx][:k]
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

	# batch write once using pandas DataFrame if available (fallback to single write)
	if dump_path and len(rows) > 0:
		dirname = os.path.dirname(dump_path)
		if dirname:
			os.makedirs(dirname, exist_ok=True)
		try:
			import pandas as pd  # type: ignore
			df = pd.DataFrame(rows, columns=["anchor_id", "recommended"])  # recommended as comma-joined string
			# preserve original space-separated format: "anchor_id rec1,rec2,..."
			df.to_csv(dump_path, sep=' ', header=False, index=False)
		except Exception:
			with open(dump_path, "w", encoding="utf-8") as f:
				for anchor_id, recs_str in rows:
					f.write(f"{anchor_id} {recs_str}\n")

	den = max(1, count)
	hit_rate_at_k = hit_hits / den
	recall_at_k = recall_sum / den
	return {"hit_rate_at_k": float(hit_rate_at_k), "recall_at_k": float(recall_at_k)}


def eval_hashtags(model: NoteLLM, notes: List[dict], max_eval: int = 200, gen_max_len: int = 32) -> float:
	# naive exact-token F1 over whitespace-split tokens
	y_pred = []
	model.model.eval()
	with torch.no_grad():
		for n in notes[:max_eval]:
			# greedy generate
			inp = f"summarize and generate hashtags: {note_text(n)}"
			inputs = model.tokenizer(inp, return_tensors="pt").to(model.model.device)
			out_ids = model.model.generate(**inputs, max_length=gen_max_len)
			decoded = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
			print(inp)
			print('decoded:', decoded)
			
			pred_tokens = set(decoded.split())
			true_tokens = set(" ".join(n.get("hashtags", n.get("tags", []))).split())
			common = len(pred_tokens & true_tokens)
			precision = common / max(1, len(pred_tokens))
			recall = common / max(1, len(true_tokens))
			f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
			y_pred.append(f1)
	return float(np.mean(y_pred))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", required=False, default=None)
	parser.add_argument("--notes_path", required=False, default=None)
	parser.add_argument("--related_path", required=False, default=None)
	parser.add_argument("--ckpt", required=False, default=None)
	parser.add_argument("--k", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--encode_max_len", type=int, default=128)
	parser.add_argument("--gen_max_len", type=int, default=32)
	parser.add_argument("--max_notes", type=int, default=None)
	parser.add_argument("--max_hashtag_eval", type=int, default=200)
	parser.add_argument("--fp16", action="store_true")
	parser.add_argument("--dump_recs_path", required=False, default=None)
	parser.add_argument("--search_batch_size", type=int, default=None)
	parser.add_argument("--use_global_index", action="store_true", help="Index on global notes (train+val), query on val only")
	args = parser.parse_args()

	if args.notes_path:
		notes_val = load_notes(args.notes_path)
	else:
		assert args.data_dir is not None, "either --data_dir or --notes_path must be provided"
		notes_val = load_notes(os.path.join(args.data_dir, "notes_val.jsonl"))

	if args.related_path:
		related = load_related(args.related_path)
	else:
		assert args.data_dir is not None, "either --data_dir or --related_path must be provided"
		related = load_related(os.path.join(args.data_dir, "related.json"))

	if args.max_notes is not None and args.max_notes > 0:
		notes_val = notes_val[:args.max_notes]

	# Load global notes (train + val) if use_global_index is True
	notes_global = None
	if args.use_global_index:
		print("Loading global notes (train + val) for indexing...")
		if args.data_dir:
			# prefer CSV if present, fallback to JSONL
			notes_train_csv = os.path.join(args.data_dir, "notes_train.csv")
			notes_train_jsonl = os.path.join(args.data_dir, "notes_train.jsonl")
			notes_train_path = notes_train_csv if os.path.exists(notes_train_csv) else notes_train_jsonl
			if os.path.exists(notes_train_path):
				notes_train = load_notes(notes_train_path)
				notes_global = notes_train + notes_val
				print(f"Loaded {len(notes_train)} train notes + {len(notes_val)} val notes = {len(notes_global)} total notes for global index")
			else:
				print(f"Warning: notes_train not found at {notes_train_csv} or {notes_train_jsonl}, using only val notes")
				notes_global = notes_val
		else:
			print("Warning: --data_dir not provided, cannot load train notes, using only val notes")
			notes_global = notes_val

	model = NoteLLM("t5-small", encode_max_len=args.encode_max_len, gen_max_len=args.gen_max_len)
	if args.ckpt:
		# assign returned pretrained modules to ensure weights are loaded
		model.model = model.model.from_pretrained(args.ckpt)
		model.tokenizer = model.tokenizer.from_pretrained(args.ckpt)
	if args.fp16:
		model.model.half()
	model.to(model.model.device)
	model.model.eval()

	metrics = eval_retrieval(
		model,
		notes_val,
		related,
		k=args.k,
		batch_size=args.batch_size,
		dump_path=args.dump_recs_path,
		search_batch_size=args.search_batch_size,
		notes_global=notes_global,
	)
	#hashtag_f1 = eval_hashtags(model, notes_val, max_eval=args.max_hashtag_eval, gen_max_len=args.gen_max_len)
	#print(json.dumps({"recall_at_k": recall_at_k, "hashtag_token_f1": hashtag_f1}))
	print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
	main()
