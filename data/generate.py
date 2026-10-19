import argparse
import json
import os
import random
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

random.seed(42)

_TOPICS = [
	"travel", "food", "beauty", "fitness", "tech", "fashion", "parenting",
	"finance", "photography", "gaming", "music", "movies", "books", "pets"
]

_LOCS = [
	"singapore", "beijing", "tokyo", "newyork", "paris", "shanghai", "seoul",
	"london", "sydney", "hongkong", "bangkok"
]

_TAGS = [
	"budget", "luxury", "tips", "guide", "review", "haul", "diy", "recipe",
	"makeup", "skincare", "streetwear", "running", "yoga", "pcbuild", "iphone",
	"android", "savemoney", "investing", "cats", "dogs"
]

_CATEGORIES = [
	"travel", "lifestyle", "beauty", "tech", "sports", "finance", "pets", "food"
]


def _make_note(i: int) -> Dict:
	"""Create a synthetic note with title, content, hashtags, category."""
	topic = random.choice(_TOPICS)
	loc = random.choice(_LOCS)
	chosen_tags = random.sample(_TAGS, k=random.randint(1, 3))
	category = random.choice(_CATEGORIES)
	title = f"{topic.title()} in {loc.title()} — {random.choice(['tips','guide','notes','diary'])}"
	content = (
		f"I recently explored {loc}. This {topic} post includes {', '.join(chosen_tags)}. "
		f"Find practical advice and my honest review."
	)
	return {
		"note_id": f"n{i}",
		"title": title,
		"content": content,
		"hashtags": [f"#{t}" for t in chosen_tags + [loc, topic]],
		"category": category,
	}


def _simulate_user_sessions(note_ids: List[str], num_users: int = 800, max_len: int = 10) -> List[List[str]]:
	"""Simulate user sessions as sequences of note views to derive co-occurrence."""
	sessions: List[List[str]] = []
	for _ in range(num_users):
		length = random.randint(2, max_len)
		session = random.sample(note_ids, k=length)
		sessions.append(session)
	return sessions


def _build_cooc(sessions: List[List[str]]) -> Dict[str, Dict[str, int]]:
	cooc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
	for seq in sessions:
		for i in range(len(seq)):
			for j in range(i + 1, len(seq)):
				a, b = seq[i], seq[j]
				cooc[a][b] += 1
				cooc[b][a] += 1
	return cooc


def _topk_related(cooc: Dict[str, Dict[str, int]], k: int = 5) -> Dict[str, List[str]]:
	related: Dict[str, List[str]] = {}
	for n, neigh in cooc.items():
		sorted_neigh = sorted(neigh.items(), key=lambda x: (-x[1], x[0]))
		related[n] = [m for m, _ in sorted_neigh[:k]]
	return related


def write_jsonl(path: str, rows: List[Dict]):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_notes_csv(path: str, rows: List[Dict]):
	"""
	Write notes to CSV with columns: note_id, content, hashtags
	- hashtags 序列会以 JSON 数组字符串形式存储，方便后续解析。
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["note_id", "content", "hashtags"])
		for r in rows:
			note_id = r.get("note_id", r.get("id", ""))
			content = r.get("content", r.get("text", ""))
			hashtags = r.get("hashtags", [])
			if not isinstance(hashtags, str):
				hashtags_str = json.dumps(hashtags, ensure_ascii=False)
			else:
				hashtags_str = hashtags
			writer.writerow([note_id, content, hashtags_str])


def write_pairs_csv(path: str, pairs: List[Dict]):
	"""
	Write note pairs to CSV with columns: note_id, pos_note_id
	其中：
	- note_id: anchor note id
	- pos_note_id: positive / co-occurred note id
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["note_id", "pos_note_id"])
		for p in pairs:
			anchor = p.get("anchor")
			positive = p.get("positive")
			writer.writerow([anchor, positive])


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--out", dest="out_dir", required=True)
	parser.add_argument("--num_notes", type=int, default=5000)
	parser.add_argument("--train_ratio", type=float, default=0.9)
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)

	notes = [_make_note(i) for i in range(args.num_notes)]
	note_ids = [n["note_id"] for n in notes]

	sessions = _simulate_user_sessions(note_ids)
	cooc = _build_cooc(sessions)
	related = _topk_related(cooc, k=5)

	# Build pairs for contrastive learning
	pairs: List[Dict] = []
	for n in note_ids:
		for m in related.get(n, [])[:3]:
			pairs.append({"anchor": n, "positive": m})

	# Split train/val
	random.shuffle(notes)
	split = int(len(notes) * args.train_ratio)
	train_notes, val_notes = notes[:split], notes[split:]

	# JSONL 输出（保持兼容）
	write_jsonl(os.path.join(args.out_dir, "notes_train.jsonl"), train_notes)
	write_jsonl(os.path.join(args.out_dir, "notes_val.jsonl"), val_notes)
	write_jsonl(os.path.join(args.out_dir, "pairs.jsonl"), pairs)

	# CSV 输出（新的规范）
	write_notes_csv(os.path.join(args.out_dir, "notes_train.csv"), train_notes)
	write_notes_csv(os.path.join(args.out_dir, "notes_val.csv"), val_notes)
	write_pairs_csv(os.path.join(args.out_dir, "note_pairs.csv"), pairs)

	# Also dump a compact mapping for quick lookup
	with open(os.path.join(args.out_dir, "related.json"), "w", encoding="utf-8") as f:
		json.dump(related, f, ensure_ascii=False)

	print(f"Wrote notes and pairs to {args.out_dir}")


if __name__ == "__main__":
	main()
