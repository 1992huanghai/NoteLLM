import argparse
import json
import os
import csv
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
import time
import math

from .modeling import NoteLLM
from .streaming import SQLiteNoteStore, iter_pairs_file, iter_sequences_file


def _parse_relaxed_jsonl_line(line: str):
	"""Parse a line that may be:
	- proper JSON (object/array)
	- CSV-quoted JSON with doubled quotes, e.g. "{""anchor"":1,""positive"":2}"
	- header tokens like _col0 (should be skipped)

	Returns parsed object or None if unparseable/empty/header.
	"""
	s = line.strip()
	if not s:
		return None
	# skip common header markers
	if s.lower() in {"_col0", "col0", "index"}:
		return None
	# fast path: normal JSON
	if s.startswith("{") or s.startswith("["):
		try:
			return json.loads(s)
		except Exception:
			pass
	# CSV-quoted JSON with doubled quotes
	if s.startswith('"') and s.endswith('"') and '""' in s:
		inner = s[1:-1]
		inner = inner.replace('""', '"')
		try:
			return json.loads(inner)
		except Exception:
			pass
	return None


def load_notes(path: str):
	"""
	加载 notes 数据。
	支持两种格式：
	1) CSV：列为 note_id, content, hashtags
	2) JSONL / CSV-quoted JSON：兼容原有实现
	"""
	items: List[dict] = []
	if path.endswith(".csv"):
		# 兼容有表头/无表头 CSV：
		# - 有表头：note_id, content, hashtags（或 id/text/tags 等别名）
		# - 无表头：按列序 [note_id, content, hashtags]
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
				i_text = _idx(["content", "text"])
				i_tags = _idx(["hashtags", "tags"])
			for row in reader:
				if not row:
					continue
					note_id = row[i_id] if i_id is not None and i_id < len(row) else None
					if note_id is None or str(note_id).strip() == "":
						continue
					content = row[i_text] if i_text is not None and i_text < len(row) else ""
					hashtags = row[i_tags] if i_tags is not None and i_tags < len(row) else ""
					items.append({"note_id": note_id, "content": content, "hashtags": hashtags})
			else:
				def _emit(row):
					if not row:
						return None
					note_id = row[0] if len(row) > 0 else None
					if note_id is None or str(note_id).strip() == "":
						return None
					content = row[1] if len(row) > 1 else ""
					hashtags = row[2] if len(row) > 2 else ""
					return {"note_id": note_id, "content": content, "hashtags": hashtags}

				obj = _emit(first)
				if obj is not None:
					items.append(obj)
				for row in reader:
					obj = _emit(row)
					if obj is not None:
				items.append(obj)
	else:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			obj = _parse_relaxed_jsonl_line(line)
			if obj is None:
				continue
			items.append(obj)
	return items


def load_pairs(path: str):
	"""
	加载 note_pairs 数据。
	支持两种格式：
	1) CSV：列为 note_id, pos_note_id
	2) JSONL / CSV-quoted JSON：兼容原有实现

	内部统一返回 dict 列表，至少包含 "anchor", "positive" 两个键。
	"""
	pairs: List[dict] = []
	if path.endswith(".csv"):
		# 兼容有表头/无表头 pairs CSV：
		# - 有表头：note_id,pos_note_id（或 anchor,positive 等别名）
		# - 无表头：按列序 [anchor, positive]
		def _is_valid(v):
			return v is not None and v != -1 and v != "-1" and v != "" and str(v).lower() != "null"

		with open(path, "r", encoding="utf-8", newline="") as f:
			reader = csv.reader(f)
			try:
				first = next(reader)
			except StopIteration:
				return []

			def _norm(s: str) -> str:
				return (s or "").strip().lower()

			header = [_norm(x) for x in first]
			has_header = any(h in {"note_id", "anchor"} for h in header) and any(h in {"pos_note_id", "positive"} for h in header)

			if has_header:
				def _idx(keys):
					for k in keys:
						if k in header:
							return header.index(k)
					return None

				i_a = _idx(["note_id", "anchor"])
				i_b = _idx(["pos_note_id", "positive"])
			for row in reader:
				if not row:
					continue
					anchor = row[i_a] if i_a is not None and i_a < len(row) else None
					positive = row[i_b] if i_b is not None and i_b < len(row) else None
				if not (_is_valid(anchor) and _is_valid(positive)):
					continue
				pairs.append({"anchor": anchor, "positive": positive})
			else:
				def _emit(row):
					if not row:
						return None
					anchor = row[0] if len(row) > 0 else None
					positive = row[1] if len(row) > 1 else None
					if not (_is_valid(anchor) and _is_valid(positive)):
						return None
					return {"anchor": anchor, "positive": positive}

				obj = _emit(first)
				if obj is not None:
					pairs.append(obj)
				for row in reader:
					obj = _emit(row)
					if obj is not None:
						pairs.append(obj)
	else:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			obj = _parse_relaxed_jsonl_line(line)
			if not isinstance(obj, dict):
				continue
			# filter obviously invalid ids early
			anchor = obj.get("anchor") or obj.get("a") or obj.get("left") or obj.get("source") or obj.get("note_a")
			positive = obj.get("positive") or obj.get("b") or obj.get("right") or obj.get("target") or obj.get("note_b")
			def _is_valid(v):
				return v is not None and v != -1 and v != "-1" and v != "" and str(v).lower() != "null"
			if not (_is_valid(anchor) and _is_valid(positive)):
				# keep the raw dict to allow downstream filtering too, but skip totally broken rows
				continue
			pairs.append(obj)
	return pairs


def batchify_pairs(pairs: List[dict], batch_size: int):
	for i in range(0, len(pairs), batch_size):
		yield pairs[i:i+batch_size]


def batchify_pairs_iter(pairs_iter, batch_size: int):
	"""Batchify a (anchor, positive) iterator into list[tuple[str,str]]."""
	buf: List[Tuple[str, str]] = []
	for a, b in pairs_iter:
		buf.append((a, b))
		if len(buf) >= batch_size:
			yield buf
			buf = []
	if buf:
		yield buf


def note_text(n: dict) -> str:
	text = n.get("content") or n.get("text")
	if not text:
		title = n.get("title", "")
		content = n.get("content", "")
		combo = (title + ". " + content).strip()
		text = combo if combo else ""
	return text


def note_tags(n: dict) -> List[str]:
	tags = n.get("hashtags") or n.get("tags") or []
	if isinstance(tags, str):
		# try to parse as JSON array string first to preserve multi-word tags
		try:
			parsed = json.loads(tags)
			if isinstance(parsed, list):
				tags = parsed
			else:
				# fallback: split on commas only (keep multi-word tokens)
				tags = [t.strip() for t in tags.split(",") if t.strip()]
		except Exception:
			# fallback: split on commas only
			tags = [t.strip() for t in tags.split(",") if t.strip()]
	# ensure strings
	return [str(t) for t in tags]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", required=True)
	parser.add_argument("--output_dir", required=True)
	parser.add_argument("--model_name_or_path", default="t5-small")
	parser.add_argument("--batch_size", type=int, default=32)  # 8GB-friendly
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--alpha_contrastive", type=float, default=1.0)
	parser.add_argument("--alpha_csft", type=float, default=1.0)
	parser.add_argument("--pairs_path", default=None)
	parser.add_argument("--sequences_path", default=None, help="可选：用户行为序列文件（CSV/JSONL）。提供后将用序列在线采样替代 pairs 做对比学习（阶段A：仍用 in-batch negatives）。")
	parser.add_argument("--window_size", type=int, default=2, help="序列窗口大小 w（默认 2）")
	parser.add_argument("--centers_per_seq", type=int, default=1, help="每条序列采样多少个 center（默认 1，越大样本量越大）")
	parser.add_argument("--positives_per_center", type=int, default=1, help="每个 center 采样多少个正样本（默认 1）。设为 0 表示使用窗口内全部正样本（多正样本展开）。")
	parser.add_argument("--seq_seed", type=int, default=42, help="序列采样随机种子（默认 42）")
	parser.add_argument("--negatives_per_pos", type=int, default=0, help="显式负采样数量（默认 0：关闭，继续用 in-batch negatives）。>0 则启用 sampled negatives loss。")
	parser.add_argument("--neg_sampling", choices=["uniform", "freq"], default="uniform", help="显式负采样策略：uniform 或 freq(按freq^0.75)。默认 uniform。")
	parser.add_argument("--neg_freq_cache", default=None, help="可选：负采样频率缓存路径（json）。不提供则在 main_process 扫描 sequences 统计并写到 output_dir/neg_freq.json")
	parser.add_argument("--encode_max_len", type=int, default=128)
	parser.add_argument("--gen_max_len", type=int, default=32)
	parser.add_argument("--csft_sub_batch", type=int, default=4)
	parser.add_argument("--gradient_checkpointing", action="store_true")
	parser.add_argument("--log_file", default=None)
	parser.add_argument("--streaming", action="store_true", help="启用 streaming 训练：pairs 流式读取，notes 用 sqlite 索引按 note_id 查")
	parser.add_argument("--notes_path", default=None, help="可选：显式指定 notes_train 文件路径（.csv 或 .jsonl）。未指定则从 data_dir 自动探测")
	parser.add_argument("--notes_db_path", default=None, help="可选：notes sqlite 索引路径。默认写到 output_dir/notes.sqlite")
	parser.add_argument("--rebuild_notes_db", action="store_true", help="重建 notes sqlite 索引（当 notes 文件更新时使用）")
	args = parser.parse_args()
	print(f"args: {args}", flush=True)
	accelerator = Accelerator()
	device = accelerator.device

	os.makedirs(args.output_dir, exist_ok=True)

	# notes 路径探测（支持显式 --notes_path）
	print(f"args.notes_path: {args.notes_path}", flush=True)
	if args.notes_path:
		notes_path = args.notes_path
	else:
		# 优先使用 CSV（notes_train.csv），否则回退到原来的 JSONL（notes_train.jsonl）
		default_notes_csv = os.path.join(args.data_dir, "notes_train.csv")
		default_notes_jsonl = os.path.join(args.data_dir, "notes_train.jsonl")
		if os.path.exists(default_notes_csv):
			notes_path = default_notes_csv
		elif os.path.exists(default_notes_jsonl):
			notes_path = default_notes_jsonl
		else:
			raise FileNotFoundError(f"notes_train not found in {args.data_dir}, expected one of: notes_train.csv or notes_train.jsonl")
	print(f"notes_path: {notes_path}", flush=True)
	# pairs: 如显式指定 --pairs_path，则按后缀自动识别；否则在 data_dir 中优先找 note_pairs.csv，其次 pairs.jsonl
	pairs_file = None
	if args.sequences_path:
		if not os.path.exists(args.sequences_path):
			raise FileNotFoundError(f"sequences file not found: {args.sequences_path}")
		print(f"[sequence] will sample contrastive pairs from sequences: {args.sequences_path}", flush=True)
	else:
	if args.pairs_path:
		pairs_file = args.pairs_path
	else:
		default_pairs_csv = os.path.join(args.data_dir, "note_pairs.csv")
		default_pairs_jsonl = os.path.join(args.data_dir, "pairs.jsonl")
		if os.path.exists(default_pairs_csv):
			pairs_file = default_pairs_csv
		else:
			pairs_file = default_pairs_jsonl
	if not os.path.exists(pairs_file):
		raise FileNotFoundError(f"pairs file not found: {pairs_file}. Pass --pairs_path to point to real_data/note_pairs.csv 或 pairs.jsonl")
	# 非 streaming：一次性读入内存（小数据集）
	pairs: Optional[List[dict]] = None
	notes_train: Optional[List[dict]] = None
	if args.streaming:
		# streaming: do NOT load notes/pairs into memory
		if pairs_file is not None:
			print(f"[streaming] pairs will be streamed from: {pairs_file} (skip load_pairs)", flush=True)
		else:
			print(f"[streaming] contrastive pairs will be sampled from sequences (skip load_pairs)", flush=True)
	else:
		print(f"[non-streaming] loading notes into memory: {notes_path}", flush=True)
		notes_train = load_notes(notes_path)
		if pairs_file is not None:
			print(f"[non-streaming] loading pairs into memory: {pairs_file}", flush=True)
	pairs = load_pairs(pairs_file)

	model = NoteLLM(args.model_name_or_path, encode_max_len=args.encode_max_len, gen_max_len=args.gen_max_len)
	if args.gradient_checkpointing:
		model.model.gradient_checkpointing_enable()
	model.to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

	# streaming 模式：使用 sqlite 索引替代 id_to_note dict
	store: Optional[SQLiteNoteStore] = None
	csft_batches_iter = None
	id_to_note = None
	neg_sampler = None
	if args.streaming:
		db_path = args.notes_db_path or os.path.join(args.output_dir, "notes.sqlite")
		store = SQLiteNoteStore(db_path)
		if accelerator.is_main_process:
			print(f"[streaming] building notes sqlite index: notes_path={notes_path} db_path={db_path} rebuild={args.rebuild_notes_db}", flush=True)
			t0 = time.time()
			store.ensure_built(notes_path, rebuild=args.rebuild_notes_db)
			print(f"[streaming] notes sqlite ready: db_path={db_path} elapsed_s={time.time()-t0:.2f}", flush=True)
		# ensure other processes wait until db exists
		accelerator.wait_for_everyone()
		# CSFT sampler (may be skipped if alpha_csft==0)
		csft_batches_iter = store.iter_cycle_batches(max(0, int(args.csft_sub_batch)))
	else:
		if notes_train is None:
			raise ValueError("notes_train is None in non-streaming mode. Check that notes_train.csv/jsonl exists and is readable, or run with --streaming.")
	id_to_note = {n.get("note_id", n.get("id", str(i))): n for i, n in enumerate(notes_train)}

	global_step = 0

	# Build negative sampler (only for streaming + sequences + explicit negatives)
	if args.streaming and args.sequences_path and int(args.negatives_per_pos) > 0:
		assert store is not None

		# Build candidate id list from sqlite once (main process) and broadcast via filesystem (simplest)
		ids_cache_path = os.path.join(args.output_dir, "notes_ids.txt")
		if accelerator.is_main_process:
			if not os.path.exists(ids_cache_path) or os.path.getsize(ids_cache_path) == 0:
				# dump note_ids once
				import sqlite3
				conn = sqlite3.connect(db_path, timeout=60)
				try:
					cur = conn.cursor()
					cur.execute("SELECT note_id FROM notes")
					with open(ids_cache_path, "w", encoding="utf-8") as f:
						for (nid,) in cur:
							f.write(str(nid) + "\n")
				finally:
					conn.close()
		accelerator.wait_for_everyone()

		with open(ids_cache_path, "r", encoding="utf-8") as f:
			neg_ids_all = [line.strip() for line in f if line.strip()]

		if not neg_ids_all:
			raise RuntimeError("negative sampler: notes_ids is empty; sqlite may be empty or not built correctly")

		if args.neg_sampling == "freq":
			freq_path = args.neg_freq_cache or os.path.join(args.output_dir, "neg_freq.json")
			if accelerator.is_main_process:
				if not os.path.exists(freq_path) or os.path.getsize(freq_path) == 0:
					# count item frequency from sequences (streaming scan)
					from collections import Counter
					c = Counter()
					for seq in iter_sequences_file(args.sequences_path):
						c.update(seq)
					# keep only ids that exist in sqlite
					c2 = {nid: int(c.get(nid, 0)) for nid in neg_ids_all}
					with open(freq_path, "w", encoding="utf-8") as wf:
						json.dump(c2, wf)
			accelerator.wait_for_everyone()
			with open(freq_path, "r", encoding="utf-8") as rf:
				freq_map: Dict[str, int] = json.load(rf)
			# build weights freq^0.75, avoid zeros
			weights = []
			for nid in neg_ids_all:
				w = float(freq_map.get(nid, 0))
				weights.append((w ** 0.75) if w > 0 else 0.0)
			# if all zeros, fallback to uniform
			if sum(weights) <= 0:
				weights = None
		else:
			weights = None

		import random
		neg_rng = random.Random(int(args.seq_seed) + 99991)

		def _sample_negs(exclude: set, k: int) -> List[str]:
			if k <= 0:
				return []
			out: List[str] = []
			# simple rejection sampling
			tries = 0
			while len(out) < k and tries < k * 50:
				tries += 1
				if weights is None:
					nid = neg_rng.choice(neg_ids_all)
				else:
					# weighted choice
					nid = neg_rng.choices(neg_ids_all, weights=weights, k=1)[0]
				if nid in exclude:
					continue
				out.append(nid)
			# if underfilled (exclude too big), allow duplicates
			if len(out) < k:
				while len(out) < k:
					nid = neg_rng.choice(neg_ids_all)
					if nid in exclude:
						continue
					out.append(nid)
			return out

		neg_sampler = _sample_negs

	for epoch in range(args.epochs):
		epoch_start_time = time.time()
		model.train()
		epoch_steps = 0
		# interval timing for every 50 steps
		interval_start_time = time.time()
		interval_start_step = global_step
		# Iterate contrastive pairs (pairs_file OR sequences_path)
		if args.sequences_path:
			import random
			rng = random.Random(int(args.seq_seed) + int(epoch))

			def _pairs_from_sequences():
				for seq in iter_sequences_file(args.sequences_path):
					# seq: list[str]
					L = len(seq)
					if L < 2:
						continue
					cps = max(1, int(args.centers_per_seq))
					w = max(1, int(args.window_size))
					ppc = int(args.positives_per_center)
					# sample centers
					if L <= cps:
						centers = list(range(L))
					else:
						centers = rng.sample(range(L), cps)
					for ci in centers:
						center = seq[ci]
						left = max(0, ci - w)
						right = min(L, ci + w + 1)
						cands = [seq[j] for j in range(left, right) if j != ci and seq[j] != center]
						if not cands:
							continue
						# multi-positive: either sample N distinct positives or emit all
						if ppc == 0:
							for pos in cands:
								yield str(center), str(pos)
						else:
							n = min(len(cands), max(1, ppc))
							# sample without replacement
							for pos in rng.sample(cands, n):
								yield str(center), str(pos)

			pairs_iter = _pairs_from_sequences()
			batches = batchify_pairs_iter(pairs_iter, args.batch_size)
		else:
			# fallback to old pairs
			if args.streaming:
				assert store is not None
				assert pairs_file is not None
				pairs_iter = iter_pairs_file(pairs_file)
			batches = batchify_pairs_iter(pairs_iter, args.batch_size)
		else:
			assert pairs is not None
			batches = batchify_pairs(pairs, args.batch_size)

		for batch in batches:
			# Filter pairs to keep one-to-one alignment
			valid_pairs: List[Tuple[str, str]] = []
			if args.streaming:
				# batch is List[Tuple[str,str]]
				# Batch fetch notes to reduce per-pair sqlite overhead
				assert store is not None
				ids: List[str] = []
				for a, b in batch:
					if a == b:
						continue
					ids.append(a)
					ids.append(b)
				id_to_note_batch = store.get_many(ids)

				notes_a: List[dict] = []
				notes_b: List[dict] = []
				for a, b in batch:
					if a == b:
						continue
					na = id_to_note_batch.get(a)
					nb = id_to_note_batch.get(b)
					if na is None or nb is None:
						continue
					valid_pairs.append((a, b))
					notes_a.append(na)
					notes_b.append(nb)
				if not valid_pairs:
					continue
				texts_a = [note_text(n) for n in notes_a]
				texts_b = [note_text(n) for n in notes_b]
			else:
				assert id_to_note is not None
			def _pid(p):
				return str(p.get("anchor") or p.get("a") or p.get("left") or p.get("source") or p.get("note_a"))

			def _qid(p):
				return str(p.get("positive") or p.get("b") or p.get("right") or p.get("target") or p.get("note_b"))

			for p in batch:
				a = _pid(p)
				b = _qid(p)
				if a in id_to_note and b in id_to_note and a != b:
					valid_pairs.append((a, b))
			if not valid_pairs:
				continue
			texts_a = [note_text(id_to_note[a]) for a, _ in valid_pairs]
			texts_b = [note_text(id_to_note[b]) for _, b in valid_pairs]

			# Contrastive loss: in-batch negatives (default) or sampled negatives
			if args.streaming and args.sequences_path and int(args.negatives_per_pos) > 0:
				# Performance: encode (a,b,negs) in ONE forward to reduce tokenizer/encoder overhead.
				assert neg_sampler is not None
				assert store is not None
				B = len(valid_pairs)
				Kneg = int(args.negatives_per_pos)
				# sample negatives per pair based on ids (exclude anchor+positive)
				all_neg_ids: List[str] = []
				neg_ids_per_pair: List[List[str]] = []
				for a_id, b_id in valid_pairs:
					excl = {a_id, b_id}
					nids = neg_sampler(excl, Kneg)
					neg_ids_per_pair.append(nids)
					all_neg_ids.extend(nids)
				# fetch negatives notes and build texts
				id_to_note_neg = store.get_many(all_neg_ids)
				neg_texts: List[str] = []
				for i in range(B):
					for nid in neg_ids_per_pair[i]:
						note = id_to_note_neg.get(nid)
						neg_texts.append(note_text(note) if note is not None else "")

				# encode all: [B + B + B*Kneg, D]
				all_texts = texts_a + texts_b + neg_texts
				emb_all = model.encode_notes(all_texts, device)
				emb_a = emb_all[:B]
				emb_b = emb_all[B : 2 * B]
				emb_neg_flat = emb_all[2 * B :]
				# reshape negatives: [B*Kneg, D] -> [B, Kneg, D]
				Ddim = emb_neg_flat.size(-1)
				emb_neg = emb_neg_flat.view(B, Kneg, Ddim)
				# scores: pos and negs
				# emb_a, emb_b: [B,D]
				s_pos = (emb_a * emb_b).sum(dim=-1, keepdim=True) / 0.07  # [B,1]
				s_neg = torch.bmm(emb_neg, emb_a.unsqueeze(-1)).squeeze(-1) / 0.07  # [B,Kneg]
				logits = torch.cat([s_pos, s_neg], dim=1)  # [B,1+Kneg]
				target = torch.zeros(B, dtype=torch.long, device=logits.device)
				loss_contrast = torch.nn.functional.cross_entropy(logits, target)
			else:
			emb_a = model.encode_notes(texts_a, device)
			emb_b = model.encode_notes(texts_b, device)
			loss_contrast = model.contrastive_loss(emb_a, emb_b)

			# CSFT on a small slice of notes (same step)
			loss_csft = torch.tensor(0.0, device=device)
			if args.alpha_csft > 0:
			if args.streaming:
				assert csft_batches_iter is not None
				subset = next(csft_batches_iter)
			else:
					if notes_train is None:
						raise ValueError("notes_train is None in non-streaming mode (CSFT). Check notes_train.csv/jsonl or run with --streaming.")
			start = (global_step * args.csft_sub_batch) % max(1, len(notes_train))
			end = min(len(notes_train), start + args.csft_sub_batch)
			subset = notes_train[start:end]
			if len(subset) > 0:
				texts = [note_text(n) for n in subset]
				tags = [note_tags(n) for n in subset]
				loss_csft = model.forward_hashtag_loss(texts, tags, device)

			loss = args.alpha_contrastive * loss_contrast + args.alpha_csft * loss_csft
			optimizer.zero_grad()
			accelerator.backward(loss)
			optimizer.step()

			global_step += 1
			epoch_steps += 1
			if accelerator.is_main_process and global_step % 50 == 0:
				interval_s = time.time() - interval_start_time
				interval_steps = max(1, global_step - interval_start_step)
				avg_step_s = interval_s / interval_steps
				msg = (
					f"epoch {epoch} step {global_step} loss {loss.item():.4f} "
					f"(contrast {loss_contrast.item():.4f} csft {loss_csft.item():.4f}) "
					f"| last_{interval_steps}_steps_s {interval_s:.2f} avg_step_s {avg_step_s:.4f}"
				)
				print(msg, flush=True)
				if args.log_file:
					dirname = os.path.dirname(args.log_file)
					if dirname:
						os.makedirs(dirname, exist_ok=True)
					with open(args.log_file, "a", encoding="utf-8") as f:
						f.write(msg + "\n")
						f.flush()
				interval_start_time = time.time()
				interval_start_step = global_step

		# Save
		if accelerator.is_main_process:
			ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
			os.makedirs(ckpt_dir, exist_ok=True)
			model.model.save_pretrained(ckpt_dir)
			model.tokenizer.save_pretrained(ckpt_dir)

		# Epoch summary timing (main process)
		epoch_total_s = time.time() - epoch_start_time
		if accelerator.is_main_process:
			avg_step_s = (epoch_total_s / max(1, epoch_steps))
			print(f"epoch {epoch} finished: steps {epoch_steps} epoch_time_s {epoch_total_s:.2f} avg_step_s {avg_step_s:.4f}", flush=True)

	if accelerator.is_main_process:
		print("Training finished.", flush=True)


if __name__ == "__main__":
	main()
