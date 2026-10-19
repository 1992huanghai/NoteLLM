import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple


SEP_ACTION = "@@@"


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
		pass


def _dedup_consecutive(items: List[str]) -> List[str]:
	"""去掉连续重复元素。"""
	deduped: List[str] = []
	prev = None
	for it in items:
		if prev is not None and it == prev:
			continue
		deduped.append(it)
		prev = it
	return deduped


def parse_action(action: str) -> Optional[Tuple[str, int]]:
	"""
	action 格式：feed_id_timestamp（用 '_' 拼接）
	注意：feed_id 自身可能含 '_'，所以用 rsplit('_', 1)。
	返回 (feed_id, ts_int)；解析失败返回 None。
	"""
	s = action.strip()
	if not s or "_" not in s:
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


def clean_items_from_seq(
	seq: str,
	sort_by_ts: bool = False,
	dedup_consecutive: bool = True,
	min_len: int = 3,
	max_len: int = 200,
) -> List[str]:
	"""把一行 behavior_seq 清洗成 item_id 序列。"""
	actions = [a for a in seq.split(SEP_ACTION) if a.strip()]
	parsed: List[Tuple[str, int]] = []
	for a in actions:
		p = parse_action(a)
		if p is None:
			continue
		parsed.append(p)
	if not parsed:
		return []
	if sort_by_ts:
		parsed.sort(key=lambda x: x[1])
	items = [fid for fid, _ in parsed]
	if dedup_consecutive:
		items = _dedup_consecutive(items)
	if max_len > 0 and len(items) > max_len:
		items = items[:max_len]
	if min_len > 0 and len(items) < min_len:
		return []
	return items


def main():
	parser = argparse.ArgumentParser(
		description="Convert user_trace.csv (user_uuid, behavior_seq) to user_sequences.csv (user_uuid, item_seq)."
	)
	parser.add_argument("--input", required=True, help="user_trace.csv path (columns: user_uuid, behavior_seq)")
	parser.add_argument("--out", required=True, help="output user_sequences.csv path")
	parser.add_argument("--user_col", default="user_uuid", help="user id column name (default: user_uuid)")
	parser.add_argument("--seq_col", default="behavior_seq", help="sequence column name (default: behavior_seq)")
	parser.add_argument("--no_header", action="store_true", help="input CSV has no header row; use column indices")
	parser.add_argument("--user_col_idx", type=int, default=0, help="user column index when --no_header (default: 0)")
	parser.add_argument("--seq_col_idx", type=int, default=1, help="seq column index when --no_header (default: 1)")
	parser.add_argument("--sort_by_ts", action="store_true", help="sort actions by timestamp before output (default: off)")
	parser.add_argument("--no_dedup_consecutive", action="store_true", help="do not deduplicate consecutive identical items")
	parser.add_argument("--min_len", type=int, default=3, help="drop sequences shorter than this (default: 3)")
	parser.add_argument("--max_len", type=int, default=200, help="truncate sequences longer than this (default: 200)")
	parser.add_argument("--max_rows", type=int, default=None, help="debug: limit number of input rows")
	args = parser.parse_args()

	_raise_csv_field_size_limit()
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

	total_rows = 0
	kept = 0
	with open(args.input, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8", newline="") as f_out:
		w = csv.writer(f_out)
		w.writerow(["user_id", "item_seq", "len"])

		if args.no_header:
			reader = csv.reader(f_in)
		else:
			reader = csv.DictReader(f_in)
			# 兼容：如果列名不一致，则尝试自动选择前两列
			if args.user_col not in (reader.fieldnames or []) or args.seq_col not in (reader.fieldnames or []):
				fns = reader.fieldnames or []
				if len(fns) >= 2:
					args.user_col = fns[0]
					args.seq_col = fns[1]
				else:
					raise ValueError(f"CSV header must include at least 2 columns. got: {fns}")

		for row in reader:
			total_rows += 1
			if args.max_rows is not None and total_rows > args.max_rows:
				break

			if args.no_header:
				if not isinstance(row, list) or len(row) <= max(args.user_col_idx, args.seq_col_idx):
					continue
				user_id = (row[args.user_col_idx] or "").strip()
				seq_str = (row[args.seq_col_idx] or "").strip()
			else:
				user_id = (row.get(args.user_col) or "").strip()
				seq_str = (row.get(args.seq_col) or "").strip()

			if not user_id or not seq_str:
				continue

			items = clean_items_from_seq(
				seq_str,
				sort_by_ts=args.sort_by_ts,
				dedup_consecutive=(not args.no_dedup_consecutive),
				min_len=args.min_len,
				max_len=args.max_len,
			)
			if not items:
				continue

			w.writerow([user_id, ",".join(items), len(items)])
			kept += 1

			if total_rows % 100_000 == 0:
				print(f"[user_trace_to_sequences] processed_rows={total_rows} kept_sequences={kept}")

	print(f"[user_trace_to_sequences] done. processed_rows={total_rows} kept_sequences={kept} out={args.out}")


if __name__ == "__main__":
	main()


