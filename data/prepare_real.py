import argparse
import json
import os
import random
import gzip
import csv
from io import StringIO
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterator, TextIO

random.seed(42)


def _is_gzip(path: str) -> bool:
	try:
		with open(path, 'rb') as f:
			head = f.read(2)
		return head == b'\x1f\x8b'
	except Exception:
		return False


def _open_text(path: str) -> TextIO:
	"""Open text file; auto-detect gzip and UTF-8 BOM (utf-8-sig)."""
	if _is_gzip(path) or path.endswith('.gz'):
		return gzip.open(path, 'rt', encoding='utf-8')
	# try utf-8 first; fall back to utf-8-sig
	try:
		return open(path, 'r', encoding='utf-8')
	except UnicodeError:
		return open(path, 'r', encoding='utf-8-sig')


def _first_non_ws_char(path: str) -> Optional[str]:
	with _open_text(path) as f:
		while True:
			ch = f.read(1)
			if not ch:
				return None
			if not ch.isspace():
				return ch


def read_jsonl(path: str) -> Iterator[dict]:
	bad = 0
	with _open_text(path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			if line == '_col0':
				# CSV-like header line
				continue
			# CSV-quoted JSON line detection
			if line.startswith('"') and line.endswith('"'):
				try:
					row = next(csv.reader([line]))
					inner = row[0]
					yield json.loads(inner)
					continue
				except Exception:
					pass
			try:
				yield json.loads(line)
			except json.JSONDecodeError:
				bad += 1
				if bad <= 5:
					print(f"[prepare_real] Skipping malformed line in {os.path.basename(path)} (count={bad})")
				continue
	if bad:
		print(f"[prepare_real] Total malformed lines skipped: {bad}")


def read_json_array_stream(path: str) -> Iterator[dict]:
	"""Stream parse a JSON array of objects without loading whole file."""
	with _open_text(path) as f:
		# seek to first '['
		ch = f.read(1)
		while ch and ch.isspace():
			ch = f.read(1)
		if ch != '[':
			# not an array
			return
		in_string = False
		escape = False
		depth = 0
		buf: List[str] = []
		while True:
			ch = f.read(1)
			if not ch:
				break
			if in_string:
				buf.append(ch)
				if escape:
					escape = False
				elif ch == '\\':
					escape = True
				elif ch == '"':
					in_string = False
				continue
			if ch == '"':
				in_string = True
				buf.append(ch)
				continue
			if ch == '{':
				depth += 1
				buf.append(ch)
				continue
			if ch == '}':
				depth -= 1
				buf.append(ch)
				if depth == 0 and buf:
					obj_text = ''.join(buf).strip()
					buf = []
					try:
						obj = json.loads(obj_text)
						yield obj
					except json.JSONDecodeError:
						obj_text2 = obj_text.rstrip(',\r\n\t ')
						try:
							obj = json.loads(obj_text2)
							yield obj
						except json.JSONDecodeError:
							pass
				continue
			if depth > 0:
				buf.append(ch)


def iter_records_any(path: str) -> Iterator[dict]:
	first = _first_non_ws_char(path)
	if first == '[':
		yield from read_json_array_stream(path)
	else:
		yield from read_jsonl(path)


def write_jsonl(path: str, rows: List[dict]):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_notes_stream(notes_path: str, out_dir: str, train_ratio: float = 0.9):
	"""Stream-split notes into train/val approximately by ratio without loading all into RAM."""
	train_path = os.path.join(out_dir, "notes_train.jsonl")
	val_path = os.path.join(out_dir, "notes_val.jsonl")
	os.makedirs(out_dir, exist_ok=True)
	count_train = 0
	count_val = 0
	with open(train_path, 'w', encoding='utf-8') as ft, open(val_path, 'w', encoding='utf-8') as fv:
		wrote_any = False
		for rec in iter_records_any(notes_path):
			bucket = ft if random.random() < train_ratio else fv
			bucket.write(json.dumps(rec, ensure_ascii=False) + "\n")
			wrote_any = True
			if bucket is ft:
				count_train += 1
			else:
				count_val += 1
	if not (count_train or count_val):
		print(f"[prepare_real] WARNING: no valid records parsed from {notes_path}. Check format (JSONL, JSON array, or CSV-quoted JSON).")
	print(f"Wrote {count_train} train and {count_val} val notes to {out_dir}")


def _extract_pair(rec: dict) -> Optional[Tuple[str, str]]:
	# tolerant to different key names
	if "anchor" in rec and "positive" in rec:
		return rec["anchor"], rec["positive"]
	cands = [
		("a", "b"), ("left", "right"), ("source", "target"), ("note_a", "note_b"),
	]
	for k1, k2 in cands:
		if k1 in rec and k2 in rec:
			return rec[k1], rec[k2]
	return None


def build_related_from_pairs(pairs_path: str, out_dir: str, topk: int = 5, bidirectional: bool = True):
	cooc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
	count = 0
	for rec in iter_records_any(pairs_path):
		p = _extract_pair(rec)
		if not p:
			continue
		a, b = p
		if not a or not b or a == b:
			continue
		cooc[a][b] += 1
		if bidirectional:
			cooc[b][a] += 1
		count += 1
	print(f"Aggregated {count} pairs into co-occurrence map for {len(cooc)} anchors")

	related: Dict[str, List[str]] = {}
	for n, neigh in cooc.items():
		sorted_neigh = sorted(neigh.items(), key=lambda x: (-x[1], x[0]))
		related[n] = [m for m, _ in sorted_neigh[:topk]]

	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join(out_dir, "related.json"), "w", encoding="utf-8") as f:
		json.dump(realted if False else related, f, ensure_ascii=False)
	print(f"Wrote related.json with top{topk} neighbors per anchor to {out_dir}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--in_dir", required=True)
	parser.add_argument("--out_dir", required=True)
	parser.add_argument("--train_ratio", type=float, default=0.9)
	parser.add_argument("--topk", type=int, default=5)
	args = parser.parse_args()

	notes_path = os.path.join(args.in_dir, "notes.jsonl")
	pairs_path = os.path.join(args.in_dir, "pairs.jsonl")
	os.makedirs(args.out_dir, exist_ok=True)

	# 1) split notes (streaming)
	split_notes_stream(notes_path, args.out_dir, train_ratio=args.train_ratio)
	# 2) build related
	build_related_from_pairs(pairs_path, args.out_dir, topk=args.topk)


if __name__ == "__main__":
	main()
