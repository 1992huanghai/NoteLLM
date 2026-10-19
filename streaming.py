import csv
import json
import os
import sqlite3
from typing import Dict, Iterator, List, Optional, Tuple


def _parse_relaxed_jsonl_line(line: str) -> Optional[object]:
	"""Parse a line that may be:
	- proper JSON (object/array)
	- CSV-quoted JSON with doubled quotes, e.g. "{""anchor"":1,""positive"":2}"
	- header tokens like _col0 (should be skipped)
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


def iter_notes_file(notes_path: str) -> Iterator[dict]:
	"""Iter notes from CSV(note_id,content,hashtags) or JSONL."""
	if notes_path.endswith(".csv"):
		# 兼容两种 CSV：
		# 1) 有表头：note_id, content, hashtags（或 id/text/tags 等别名）
		# 2) 无表头：按列序 [note_id, content, hashtags]
		with open(notes_path, "r", encoding="utf-8", newline="") as f:
			reader = csv.reader(f)
			try:
				first = next(reader)
			except StopIteration:
				return

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
					yield {"note_id": str(note_id), "content": content or "", "hashtags": hashtags or ""}
			else:
				def _emit(row):
					if not row:
						return
					note_id = row[0] if len(row) > 0 else None
					if note_id is None or str(note_id).strip() == "":
						return
					content = row[1] if len(row) > 1 else ""
					hashtags = row[2] if len(row) > 2 else ""
					yield {"note_id": str(note_id), "content": content or "", "hashtags": hashtags or ""}

				for obj in _emit(first):
					yield obj
				for row in reader:
					for obj in _emit(row):
						yield obj
	else:
		with open(notes_path, "r", encoding="utf-8") as f:
			for line in f:
				obj = _parse_relaxed_jsonl_line(line)
				if not isinstance(obj, dict):
					continue
				note_id = obj.get("note_id", obj.get("id"))
				if note_id is None or str(note_id).strip() == "":
					continue
				yield obj


def iter_pairs_file(pairs_path: str) -> Iterator[Tuple[str, str]]:
	"""Iter (anchor_id, positive_id) from CSV(note_id,pos_note_id) or JSONL-like."""
	def _is_valid(v) -> bool:
		return v is not None and v != -1 and v != "-1" and v != "" and str(v).lower() != "null"

	if pairs_path.endswith(".csv"):
		# 兼容有表头/无表头的 pairs CSV：
		# - 有表头：note_id,pos_note_id（或 anchor,positive 等别名）
		# - 无表头：按列序 [anchor, positive]
		with open(pairs_path, "r", encoding="utf-8", newline="") as f:
			reader = csv.reader(f)
			try:
				first = next(reader)
			except StopIteration:
				return

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
				if not row:
					continue
					anchor = row[i_a] if i_a is not None and i_a < len(row) else None
					positive = row[i_b] if i_b is not None and i_b < len(row) else None
				if not (_is_valid(anchor) and _is_valid(positive)):
					continue
				a = str(anchor)
				b = str(positive)
				if a == b:
					continue
				yield a, b
			else:
				def _emit(row):
					if not row:
						return
					anchor = row[0] if len(row) > 0 else None
					positive = row[1] if len(row) > 1 else None
					if not (_is_valid(anchor) and _is_valid(positive)):
						return
					a = str(anchor)
					b = str(positive)
					if a == b:
						return
					yield a, b

				for tup in _emit(first):
					yield tup
				for row in reader:
					for tup in _emit(row):
						yield tup
	else:
		with open(pairs_path, "r", encoding="utf-8") as f:
			for line in f:
				obj = _parse_relaxed_jsonl_line(line)
				if not isinstance(obj, dict):
					continue
				anchor = obj.get("anchor") or obj.get("a") or obj.get("left") or obj.get("source") or obj.get("note_a")
				positive = obj.get("positive") or obj.get("b") or obj.get("right") or obj.get("target") or obj.get("note_b")
				if not (_is_valid(anchor) and _is_valid(positive)):
					continue
				a = str(anchor)
				b = str(positive)
				if a == b:
					continue
				yield a, b


def iter_sequences_file(seqs_path: str) -> Iterator[List[str]]:
	"""
	Iter user/item sequences from CSV or JSONL.

	CSV supported:
	- With header: columns include item_seq (comma-joined) or item_ids (comma-joined)
	- Without header: treat first column as item_seq

	JSONL supported:
	- Each line: {"item_seq": "a,b,c"} or {"item_ids": ["a","b","c"]}
	"""
	def _emit_seq(items: List[str]):
		items = [str(x).strip() for x in items if str(x).strip()]
		if len(items) >= 2:
			yield items

	if seqs_path.endswith(".csv"):
		with open(seqs_path, "r", encoding="utf-8", newline="") as f:
			reader = csv.reader(f)
			try:
				first = next(reader)
			except StopIteration:
				return

			def _norm(s: str) -> str:
				return (s or "").strip().lower()

			header = [_norm(x) for x in first]
			has_header = any(h in {"item_seq", "item_ids", "seq", "sequence"} for h in header)

			if has_header:
				def _idx(keys):
					for k in keys:
						if k in header:
							return header.index(k)
					return None

				i_seq = _idx(["item_seq", "seq", "sequence"])
				i_ids = _idx(["item_ids"])

				for row in reader:
					if not row:
						continue
					if i_seq is not None and i_seq < len(row) and row[i_seq]:
						items = [x for x in row[i_seq].split(",")]
						for seq in _emit_seq(items):
							yield seq
					elif i_ids is not None and i_ids < len(row) and row[i_ids]:
						# comma-joined in CSV for simplicity
						items = [x for x in row[i_ids].split(",")]
						for seq in _emit_seq(items):
							yield seq
			else:
				# treat first row as data; first column is item_seq
				for seq in _emit_seq([x for x in first[0].split(",")] if first and len(first) > 0 else []):
					yield seq
				for row in reader:
					if not row or len(row) == 0:
						continue
					for seq in _emit_seq([x for x in row[0].split(",")]):
						yield seq
	else:
		with open(seqs_path, "r", encoding="utf-8") as f:
			for line in f:
				obj = _parse_relaxed_jsonl_line(line)
				if not isinstance(obj, dict):
					continue
				if "item_ids" in obj and isinstance(obj["item_ids"], list):
					for seq in _emit_seq(obj["item_ids"]):
						yield seq
				else:
					s = obj.get("item_seq") or obj.get("seq") or obj.get("sequence")
					if isinstance(s, str) and s.strip():
						for seq in _emit_seq([x for x in s.split(",")]):
							yield seq


class SQLiteNoteStore:
	"""
	一个非常轻量的 notes 索引：
	- 适合 300w notes 这类规模：训练阶段通过 note_id 快速查 content/hashtags
	- 使用 sqlite 文件：构建一次，之后只读查询
	"""

	def __init__(self, db_path: str):
		self.db_path = db_path
		# Keep a process-local read-only connection open to avoid per-batch connect/close overhead.
		# (Safe with accelerate multi-process since each process has its own Python memory space.)
		self._read_conn: Optional[sqlite3.Connection] = None

	def _connect_read(self) -> sqlite3.Connection:
		"""
		Fast read connection for training-time lookups.
		We intentionally keep this connection open to reduce overhead.
		"""
		if self._read_conn is not None:
			return self._read_conn
		conn = sqlite3.connect(self.db_path, timeout=60)
		conn.row_factory = sqlite3.Row
		cur = conn.cursor()
		# Read performance pragmas (do NOT use for writes).
		# - query_only: guarantees no accidental writes and can reduce locking.
		# - cache_size: negative means KB, e.g. -200000 ~= 200MB cache.
		# - mmap_size: memory-map db file for faster reads (best on local SSD).
		cur.execute("PRAGMA query_only=1;")
		cur.execute("PRAGMA temp_store=MEMORY;")
		cur.execute("PRAGMA cache_size=-200000;")
		cur.execute("PRAGMA mmap_size=268435456;")  # 256MB
		self._read_conn = conn
		return conn

	def ensure_built(self, notes_path: str, rebuild: bool = False, commit_every: int = 50_000):
		os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
		if rebuild and os.path.exists(self.db_path):
			os.remove(self.db_path)
		if os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0 and not rebuild:
			# If it's non-empty but contains 0 rows (common when CSV header mismatch), rebuild.
			try:
				conn = sqlite3.connect(self.db_path, timeout=60)
				try:
					cur = conn.cursor()
					cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notes'")
					if cur.fetchone() is None:
						# no table -> rebuild
						pass
					else:
						cur.execute("SELECT COUNT(*) FROM notes")
						n = cur.fetchone()[0]
						if n and int(n) > 0:
			return
				finally:
					conn.close()
			except Exception:
				# fall through to rebuild
				pass

		# Build connection (write path). Do NOT reuse read connection / read pragmas.
		conn = sqlite3.connect(self.db_path, timeout=60)
		conn.row_factory = sqlite3.Row
		try:
			cur = conn.cursor()
			cur.execute("PRAGMA journal_mode=WAL;")
			cur.execute("PRAGMA synchronous=OFF;")
			cur.execute("PRAGMA temp_store=MEMORY;")
			cur.execute(
				"""
				CREATE TABLE IF NOT EXISTS notes (
				  note_id TEXT PRIMARY KEY,
				  content TEXT,
				  hashtags TEXT
				);
				"""
			)
			conn.commit()

			buf: List[Tuple[str, str, str]] = []
			total = 0
			for n in iter_notes_file(notes_path):
				note_id = str(n.get("note_id", n.get("id")))
				content = n.get("content") or n.get("text") or ""
				hashtags = n.get("hashtags") or n.get("tags") or ""
				if not isinstance(hashtags, str):
					hashtags = json.dumps(hashtags, ensure_ascii=False)
				buf.append((note_id, str(content), str(hashtags)))
				if len(buf) >= commit_every:
					cur.executemany(
						"INSERT OR REPLACE INTO notes(note_id, content, hashtags) VALUES(?,?,?)",
						buf,
					)
					conn.commit()
					total += len(buf)
					buf = []
			if buf:
				cur.executemany(
					"INSERT OR REPLACE INTO notes(note_id, content, hashtags) VALUES(?,?,?)",
					buf,
				)
				conn.commit()
				total += len(buf)
		finally:
			conn.close()

	def get(self, note_id: str) -> Optional[Dict[str, str]]:
		conn = self._connect_read()
			cur = conn.cursor()
			cur.execute("SELECT note_id, content, hashtags FROM notes WHERE note_id = ?", (str(note_id),))
			row = cur.fetchone()
			if row is None:
				return None
			return {"note_id": row["note_id"], "content": row["content"], "hashtags": row["hashtags"]}

	def get_many(self, note_ids: List[str]) -> Dict[str, Dict[str, str]]:
		"""Batch fetch notes by ids. Much faster than calling get() in a Python loop."""
		ids = [str(x) for x in note_ids if x is not None]
		if not ids:
			return {}
		# SQLite has a default variable limit (often 999). Our batches are small (<= 2*batch_size).
		placeholders = ",".join(["?"] * len(ids))
		sql = f"SELECT note_id, content, hashtags FROM notes WHERE note_id IN ({placeholders})"
		conn = self._connect_read()
		cur = conn.cursor()
		cur.execute(sql, ids)
		out: Dict[str, Dict[str, str]] = {}
		for row in cur.fetchall():
			out[str(row["note_id"])] = {"note_id": row["note_id"], "content": row["content"], "hashtags": row["hashtags"]}
		return out

	def iter_cycle_batches(self, batch_size: int) -> Iterator[List[Dict[str, str]]]:
		"""无限循环地按顺序遍历整库，按 batch_size yield，适合作为 CSFT 子批次采样。"""
		if batch_size <= 0:
			while True:
				yield []
		while True:
			conn = self._connect_read()
				cur = conn.cursor()
				cur.execute("SELECT note_id, content, hashtags FROM notes")
				buf: List[Dict[str, str]] = []
				for row in cur:
					buf.append({"note_id": row["note_id"], "content": row["content"], "hashtags": row["hashtags"]})
					if len(buf) >= batch_size:
						yield buf
						buf = []
				if buf:
					yield buf


