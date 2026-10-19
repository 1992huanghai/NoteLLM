import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict, Counter
from typing import Iterable, List, Optional, Tuple


SEP_ACTION = "@@@"


def parse_action(action: str) -> Optional[Tuple[str, int]]:
	"""
	action 格式：feed_id_timestamp（用 '_' 拼接）
	注意：feed_id 自身可能含 '_'，所以用 rsplit('_', 1)。
	返回 (feed_id, ts_int)；解析失败返回 None。
	"""
	s = action.strip()
	if not s:
		return None
	if "_" not in s:
		return None
	feed_id, ts = s.rsplit("_", 1)
	feed_id = feed_id.strip()
	if not feed_id:
		return None
	try:
		ts_int = int(ts)
	except Exception:
		return None
	return feed_id, ts_int


def iter_adjacent_pairs_from_seq(seq: str, sort_by_ts: bool = False, dedup_consecutive: bool = True) -> Iterable[Tuple[str, str]]:
	"""
	从一个用户行为序列中产生相邻 pair：
	- seq: 由 '@@@' 分割的 action 串
	- sort_by_ts: 是否按时间戳排序（默认 False，认为原序列已按时间排列）
	- dedup_consecutive: 是否跳过连续重复 item（默认 True）
	"""
	actions = [a for a in seq.split(SEP_ACTION) if a.strip()]
	parsed: List[Tuple[str, int]] = []
	for a in actions:
		p = parse_action(a)
		if p is None:
			continue
		parsed.append(p)
	if not parsed or len(parsed) < 2:
		return []

	if sort_by_ts:
		parsed.sort(key=lambda x: x[1])

	items = [fid for fid, _ in parsed]
	if dedup_consecutive:
		items = _dedup_consecutive(items)

	if len(items) < 2:
		return []

	for i in range(len(items) - 1):
		a = items[i]
		b = items[i + 1]
		if not a or not b or a == b:
			continue
		yield a, b


def _raise_csv_field_size_limit():
	"""
	提升 csv.field_size_limit 到平台允许的最大值，
	避免处理超长字段时报 '_csv.Error: field larger than field limit'
	"""
	try:
		max_int = sys.maxsize
		while True:
			try:
				csv.field_size_limit(max_int)
				break
			except OverflowError:
				max_int = int(max_int / 10)
	except Exception:
		# 如果设置失败就继续，用默认限制（可能仍会报错）
		pass


def _dedup_consecutive(items: List[str]) -> List[str]:
	"""
	去掉列表中连续重复的元素，例如 ['A','A','B','A'] -> ['A','B','A']
	"""
	deduped: List[str] = []
	prev = None
	for it in items:
		if prev is not None and it == prev:
			continue
		deduped.append(it)
		prev = it
	return deduped


def _connect(db_path: str) -> sqlite3.Connection:
	conn = sqlite3.connect(db_path, timeout=60)
	conn.execute("PRAGMA journal_mode=WAL;")
	conn.execute("PRAGMA synchronous=OFF;")
	conn.execute("PRAGMA temp_store=MEMORY;")
	return conn


def build_pair_counts_sqlite(
	input_csv: str,
	db_path: str,
	user_col: str = "user_uuid",
	seq_col: str = "behavior_seq",
	sort_by_ts: bool = False,
	dedup_consecutive: bool = True,
	commit_every: int = 2_000_000,
	max_unique_in_mem: int = 2_000_000,
	max_rows: Optional[int] = None,
	no_header: bool = False,
	user_col_idx: int = 0,
	seq_col_idx: int = 1,
):
	"""
	将 user_trace.csv 聚合成 pair count 表（sqlite）。
	
	如果 no_header=True，则按列索引读取（user_col_idx, seq_col_idx）；
	否则用 DictReader 按列名（user_col, seq_col）读取。
	"""
	# user_trace 的行为序列字段可能非常长，提升 csv 单字段长度上限
	_raise_csv_field_size_limit()

	os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
	if os.path.exists(db_path):
		os.remove(db_path)
	conn = _connect(db_path)
	try:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS pairs (
			  note_id TEXT NOT NULL,
			  pos_note_id TEXT NOT NULL,
			  cnt INTEGER NOT NULL,
			  PRIMARY KEY (note_id, pos_note_id)
			);
			"""
		)
		conn.commit()

		cur = conn.cursor()
		# 批量 upsert（按批次聚合后的 count 写入），显著减少 sqlite 写放大
		upsert_sql = (
			"INSERT INTO pairs(note_id, pos_note_id, cnt) VALUES(?,?,?) "
			"ON CONFLICT(note_id, pos_note_id) DO UPDATE SET cnt = cnt + excluded.cnt"
		)

		# in-memory aggregation: (a,b) -> cnt
		counts = defaultdict(int)
		pairs_since_flush = 0
		rows = 0
		pairs_total = 0

		def flush():
			nonlocal pairs_since_flush, counts
			if not counts:
				return
			items = [(a, b, c) for (a, b), c in counts.items()]
			cur.executemany(upsert_sql, items)
			conn.commit()
			pairs_since_flush = 0
			counts.clear()

		with open(input_csv, "r", encoding="utf-8") as f:
			if no_header:
				reader = csv.reader(f)
			else:
				reader = csv.DictReader(f)
				# 兼容：如果列名不一致，则尝试自动选择第二列作为序列列
				if user_col not in (reader.fieldnames or []) or seq_col not in (reader.fieldnames or []):
					fns = reader.fieldnames or []
					if len(fns) >= 2:
						user_col = fns[0]
						seq_col = fns[1]
					else:
						raise ValueError(f"CSV header must include at least 2 columns. got: {fns}")

			for row in reader:
				rows += 1
				if max_rows is not None and rows > max_rows:
					break
				
				if no_header:
					# row 是 list，按索引取
					if len(row) <= seq_col_idx:
						continue
					seq = row[seq_col_idx].strip() if row[seq_col_idx] else ""
				else:
					# row 是 dict，按列名取
					seq = (row.get(seq_col) or "").strip()
				
				if not seq:
					continue
				for a, b in iter_adjacent_pairs_from_seq(seq, sort_by_ts=sort_by_ts, dedup_consecutive=dedup_consecutive):
					counts[(a, b)] += 1
					pairs_since_flush += 1
					pairs_total += 1
					# flush by emitted pairs or distinct pairs in memory
					if pairs_since_flush >= commit_every or len(counts) >= max_unique_in_mem:
						flush()
				if rows % 100_000 == 0:
					print(f"[user_trace_to_pairs] processed_rows={rows} emitted_pairs={pairs_total}")

		flush()

		print(f"[user_trace_to_pairs] done. processed_rows={rows} emitted_pairs={pairs_total}")
	finally:
		conn.close()


def build_pair_counts_memory(
	input_csv: str,
	user_col: str = "user_uuid",
	seq_col: str = "behavior_seq",
	sort_by_ts: bool = False,
	dedup_consecutive: bool = True,
	max_rows: Optional[int] = None,
	no_header: bool = False,
	user_col_idx: int = 0,
	seq_col_idx: int = 1,
) -> Counter:
	"""
	纯内存版本：用 Counter 聚合 pair 计数，不写 sqlite。
	返回 Counter[(note_id, pos_note_id)] = count
	
	如果 no_header=True，则按列索引读取（user_col_idx, seq_col_idx）；
	否则用 DictReader 按列名（user_col, seq_col）读取。
	"""
	pair_counter = Counter()
	_raise_csv_field_size_limit()
	with open(input_csv, "r", encoding="utf-8") as f:
		if no_header:
			reader = csv.reader(f)
		else:
			reader = csv.DictReader(f)
		
		processed_rows = 0
		for row in reader:
			if max_rows is not None and processed_rows >= max_rows:
				break
			
			if no_header:
				# row 是 list，按索引取
				if len(row) <= max(user_col_idx, seq_col_idx):
					processed_rows += 1
					continue
				seq_str = row[seq_col_idx]
			else:
				# row 是 dict，按列名取
				seq_str = row.get(seq_col, "")
			
			if not seq_str:
				processed_rows += 1
				continue
			behaviors = [b.strip() for b in seq_str.split("@@@") if b.strip()]
			parsed = []
			for b in behaviors:
				if "_" not in b:
					continue
				feed_id_str, ts_str = b.rsplit("_", 1)
				try:
					ts = int(ts_str)
				except ValueError:
					continue
				parsed.append((feed_id_str, ts))
			if sort_by_ts:
				parsed.sort(key=lambda x: x[1])
			items = [feed_id for feed_id, _ in parsed]
			if dedup_consecutive:
				items = _dedup_consecutive(items)
			for i in range(len(items) - 1):
				pair_counter[(items[i], items[i + 1])] += 1
			processed_rows += 1
			if processed_rows % 100000 == 0:
				print(f"[build_pair_counts_memory] processed_rows={processed_rows} unique_pairs={len(pair_counter)}")
	print(f"[build_pair_counts_memory] done. total_rows={processed_rows} unique_pairs={len(pair_counter)} total_count={sum(pair_counter.values())}")
	return pair_counter


def export_pairs_from_counter(
	pair_counter: Counter,
	out_csv: str,
	min_count: int = 5,
	topk_per_note: Optional[int] = None,
	with_count: bool = False,
):
	"""
	从 Counter 导出 pairs 到 CSV，支持 min_count 和 topk_per_note 过滤。
	"""
	os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
	# 先按 count 过滤
	filtered = {pair: cnt for pair, cnt in pair_counter.items() if cnt > min_count}
	print(f"[export_pairs_from_counter] after min_count={min_count} filter: {len(filtered)} pairs")
	
	# topk_per_note 过滤
	if topk_per_note is not None:
		note_neighbors = defaultdict(list)
		for (note_id, pos_note_id), cnt in filtered.items():
			note_neighbors[note_id].append((pos_note_id, cnt))
		final_pairs = []
		for note_id, neighbors in note_neighbors.items():
			neighbors.sort(key=lambda x: -x[1])
			for pos_note_id, cnt in neighbors[:topk_per_note]:
				final_pairs.append((note_id, pos_note_id, cnt))
		print(f"[export_pairs_from_counter] after topk_per_note={topk_per_note}: {len(final_pairs)} pairs")
	else:
		final_pairs = [(note_id, pos_note_id, cnt) for (note_id, pos_note_id), cnt in filtered.items()]
	
	# 写 CSV
	with open(out_csv, "w", encoding="utf-8", newline="") as f:
		w = csv.writer(f)
		if with_count:
			w.writerow(["note_id", "pos_note_id", "count"])
		else:
			w.writerow(["note_id", "pos_note_id"])
		for note_id, pos_note_id, cnt in final_pairs:
			if with_count:
				w.writerow([note_id, pos_note_id, int(cnt)])
			else:
				w.writerow([note_id, pos_note_id])
	print(f"[export_pairs_from_counter] wrote {len(final_pairs)} pairs to {out_csv}")


def export_filtered_pairs(
	db_path: str,
	out_csv: str,
	min_count: int = 5,
	topk_per_note: Optional[int] = None,
	with_count: bool = False,
):
	"""
	导出 cnt > min_count 的 pairs。
	如果指定 topk_per_note，则对每个 note_id 只保留 cnt 最高的前 K 个邻居。
	默认输出两列（note_id,pos_note_id）以匹配模型输入；
	可用 --with_count 追加第三列 cnt 便于排查。
	"""
	os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
	conn = _connect(db_path)
	try:
		cur = conn.cursor()
		if topk_per_note is None:
			# 不限制 topk：直接按 cnt > min_count 过滤
			cur.execute("SELECT note_id, pos_note_id, cnt FROM pairs WHERE cnt > ? ORDER BY cnt DESC", (int(min_count),))
		else:
			# 限制 topk：用窗口函数对每个 note_id 按 cnt 降序取前 K
			sql = """
				WITH ranked AS (
					SELECT note_id, pos_note_id, cnt,
					       ROW_NUMBER() OVER (PARTITION BY note_id ORDER BY cnt DESC) as rn
					FROM pairs
					WHERE cnt > ?
				)
				SELECT note_id, pos_note_id, cnt FROM ranked WHERE rn <= ?
				ORDER BY note_id, cnt DESC
			"""
			cur.execute(sql, (int(min_count), int(topk_per_note)))
		with open(out_csv, "w", encoding="utf-8", newline="") as f:
			w = csv.writer(f)
			if with_count:
				w.writerow(["note_id", "pos_note_id", "count"])
			else:
				w.writerow(["note_id", "pos_note_id"])
			n = 0
			for note_id, pos_note_id, cnt in cur:
				if with_count:
					w.writerow([note_id, pos_note_id, int(cnt)])
				else:
					w.writerow([note_id, pos_note_id])
				n += 1
				if n % 1_000_000 == 0:
					print(f"[user_trace_to_pairs] exported={n}")
		print(f"[user_trace_to_pairs] export done. out={out_csv}")
	finally:
		conn.close()


def main():
	parser = argparse.ArgumentParser(description="Convert user_trace.csv to note_pairs.csv (adjacent pairs) with aggregation and filtering.")
	parser.add_argument("--input", required=True, help="user_trace.csv path (columns: user_uuid, behavior_seq)")
	parser.add_argument("--out", required=True, help="output note_pairs.csv path")
	parser.add_argument("--min_count", type=int, default=5, help="keep pairs with count > min_count (default: 5)")
	parser.add_argument("--topk_per_note", type=int, default=None, help="keep only topK neighbors per note (by count DESC); None = no limit (default: None)")
	parser.add_argument("--use_memory_only", action="store_true", help="use pure in-memory Counter (much faster, but needs enough RAM)")
	parser.add_argument("--no_header", action="store_true", help="input CSV has no header row; use column indices (col0=user, col1=seq)")
	parser.add_argument("--db_path", default=None, help="sqlite path for aggregation (default: alongside --out as pairs.sqlite)")
	parser.add_argument("--user_col", default="user_uuid", help="user id column name (default: user_uuid)")
	parser.add_argument("--seq_col", default="behavior_seq", help="sequence column name (default: behavior_seq)")
	parser.add_argument("--sort_by_ts", action="store_true", help="sort actions by timestamp before pairing (default: off)")
	parser.add_argument("--no_dedup_consecutive", action="store_true", help="do not deduplicate consecutive identical items")
	parser.add_argument("--commit_every", type=int, default=2_000_000, help="flush to sqlite every N emitted pairs (default: 2000000)")
	parser.add_argument("--max_unique_in_mem", type=int, default=2_000_000, help="flush to sqlite when in-memory unique pairs exceed this (default: 2000000)")
	parser.add_argument("--max_rows", type=int, default=None, help="debug: limit number of input rows")
	parser.add_argument("--with_count", action="store_true", help="output third column 'count' for debugging")
	args = parser.parse_args()

	if args.use_memory_only:
		# 纯内存模式：用 Counter 聚合，跳过 sqlite
		print("[main] using pure in-memory mode (--use_memory_only)")
		pair_counter = build_pair_counts_memory(
			input_csv=args.input,
			user_col=args.user_col,
			seq_col=args.seq_col,
			sort_by_ts=args.sort_by_ts,
			dedup_consecutive=(not args.no_dedup_consecutive),
			max_rows=args.max_rows,
			no_header=args.no_header,
		)
		export_pairs_from_counter(
			pair_counter=pair_counter,
			out_csv=args.out,
			min_count=args.min_count,
			topk_per_note=args.topk_per_note,
			with_count=args.with_count,
		)
	else:
		# sqlite 模式（默认）
		db_path = args.db_path or (os.path.splitext(args.out)[0] + ".sqlite")
		build_pair_counts_sqlite(
			input_csv=args.input,
			db_path=db_path,
			user_col=args.user_col,
			seq_col=args.seq_col,
			sort_by_ts=args.sort_by_ts,
			dedup_consecutive=(not args.no_dedup_consecutive),
			commit_every=args.commit_every,
			max_unique_in_mem=args.max_unique_in_mem,
			max_rows=args.max_rows,
			no_header=args.no_header,
		)
		export_filtered_pairs(db_path=db_path, out_csv=args.out, min_count=args.min_count, topk_per_note=args.topk_per_note, with_count=args.with_count)


if __name__ == "__main__":
	main()


