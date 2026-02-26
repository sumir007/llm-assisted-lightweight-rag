import os
import re
import json
import csv
import time
import requests
from datetime import datetime
from uuid import uuid4
try:
import pdfplumber
PDF_SUPPORT = True
except ImportError:
PDF_SUPPORT = False


# CONFIG (EDIT LOCALLY)
GITHUB_PAT = os.getenv("GITHUB_PAT", "NON") # don't commit
MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4.1")
API_URL = "https://models.github.ai/inference/chat/completions"

# Single file or folder path
DOC_FILES = [
#os.path.join(os.path.expanduser("~"), "Desktop", "doc2.txt")
]

# Batch PDF processing
PDF_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "LLM", "PDF")
BATCH_SIZE = 1 # Process 1 PDF at a time
PDFS_PER_MINUTE = 10 # Process 10 PDFs per 60 seconds
DELAY_BETWEEN_PDFS = 60 / PDFS_PER_MINUTE # 6 seconds between each PDF

RESPONSES_LOG_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "LLM", "rag_responses_log.csv")


DELIMITER = "---"
TOP_K = 3

# LLM-ASSISTED RETRIEVAL CONFIG

ENABLE_LLM_SEARCH = True # Set to False to disable LLM-assisted query expansion
SEARCH_MODEL = os.getenv("GITHUB_SEARCH_MODEL", "openai/gpt-4o") # you can override with a cheaper model
SEARCH_MAX_TOKENS = 300 # cap for the search-expansion LLM step (not the final answer)

SEARCH_SYSTEM_PROMPT = (
"You extract search signals from a user's question to retrieve relevant document chunks.\n"
"Return STRICT JSON (no prose) with keys:\n"
" - canonical_phrases: list of exact phrases that best capture the query intent.\n"
" - synonyms: object mapping each canonical phrase to a list of common aliases/synonyms.\n"
" - keywords: list of single- or multi-word keywords that help find relevant text.\n"
"Rules:\n"
"- Be precise and conservative; include only what the question clearly implies.\n"
"- Prefer phrases that are likely to literally appear in docs (e.g., 'Date of Agreement').\n"
"- No explanation, no markdown—only valid JSON."
)

# Rough input size controls (chars ~= tokens*4)
MAX_QUESTION_CHARS = 800
MAX_CHUNK_CHARS = 2000
MAX_INPUT_CHARS = 7000

# Optional output cap (completion tokens)
MAX_COMPLETION_TOKENS = 300

TOKEN_RX = re.compile(r"[A-Za-z0-9_\-]+")

# ---- DATE & VALIDITY PATTERNS ----
MONTHS = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
ORD = r"(st|nd|rd|th)?"

DATE_RX_LIST = [
re.compile(r"\b\d{4}-\d{2}-\d{2}\b", re.I), # 2026-01-30
re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I), # 30/01/2026 or 30/1/26
re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", re.I), # 30.01.26
re.compile(rf"\b\d{{1,2}}{ORD}\s+of\s+{MONTHS}\s+\d{{4}}\b", re.I), # 30th of January 2026
re.compile(rf"\b{MONTHS}\s+\d{{1,2}}{ORD},?\s+\d{{4}}\b", re.I), # January 30, 2026
]

VALIDITY_RX_LIST = [
re.compile(r"\bvalid\s+for\s+\d+\s+(month|months|year|years)\b", re.I),
re.compile(r"\bterm\s+of\s+\d+\s+(month|months|year|years)\b", re.I),
re.compile(r"\bfor\s+a\s+period\s+of\s+\d+\s+(month|months|year|years)\b", re.I),
re.compile(r"\b(duration|validity)\b", re.I),
]

AGREEMENT_DATE_PHRASES = [
"date of agreement",
"agreement date",
"date of this agreement",
"date hereof",
]

def has_any(rx_list, text: str) -> bool:
tl = text.lower()
return any(rx.search(tl) for rx in rx_list)

def count_word_boundary_occurrences(text: str, term: str) -> int:
"""
Count occurrences of a term using word-boundaries for alphanumeric terms.
Falls back to substring for multi-word phrases.
"""
tl = text.lower()
term = (term or "").lower().strip()
if not term:
return 0
# Use \b only if term looks single-token/alphanumeric
if re.fullmatch(r"[a-z0-9\-_.]+", term):
rx = re.compile(rf"\b{re.escape(term)}\b", re.I)
return len(rx.findall(tl))
# Multi-word (or contains spaces): simple non-overlapping substring count
cnt, start = 0, 0
while True:
idx = tl.find(term, start)
if idx == -1:
break
cnt += 1
start = idx + len(term)
return cnt

SYSTEM_PROMPT = (
"You are a helpful assistant "
"Use ONLY the provided context chunks. "
"If the answer is not in the chunks, say \"I don't know\". "
"Keep it concise. "
"When you use a chunk, cite it like [Chunk 1], [Chunk 2]."
"If asked whether something has expired: reply with only the word 'Expired' if expired, "
"otherwise reply with only the word 'Valid'."
)


# KNOWLEDGE BASE
def extract_text_from_pdf(pdf_path: str) -> str:
if not PDF_SUPPORT:
print(f"[warning] pdfplumber not installed. Install with: pip install pdfplumber")
return ""

try:
with pdfplumber.open(pdf_path) as pdf:
text = ""
for page in pdf.pages:
text += page.extract_text() or ""
text += "\n" # Add newline between pages
return text
except Exception as e:
print(f"[error] Failed to extract text from {pdf_path}: {e}")
return ""


def insert_delimiters_every_n_words(text: str, n: int = 100, delimiter: str = "---") -> str:
words = text.split()
result = [delimiter]

for i, word in enumerate(words):
result.append(word)
# Add delimiter after every n words (but not after the last word)
if (i + 1) % n == 0 and i + 1 < len(words):
result.append(delimiter)

result.append(delimiter)
return " ".join(result)


def split_by_delimiter(text: str, delimiter: str):
buf = []
for line in text.splitlines():
if line.strip() == delimiter:
chunk = "\n".join(buf).strip()
buf = []
if chunk:
yield chunk
else:
buf.append(line)
last = "\n".join(buf).strip()
if last:
yield last

def get_pdf_files_from_folder(folder_path: str) -> list[str]:
if not os.path.isdir(folder_path):
print(f"[warning] PDF folder not found: {folder_path}")
return []

pdf_files = []
for filename in sorted(os.listdir(folder_path)):
if filename.lower().endswith(".pdf"):
pdf_files.append(os.path.join(folder_path, filename))

return pdf_files


def load_chunks(files: list[str] = None):
if files is None:
files = DOC_FILES

chunks = []
for path in files:
if not os.path.exists(path):
continue

txt = ""
# Check if file is a PDF
if path.lower().endswith(".pdf"):
txt = extract_text_from_pdf(path)
else:
# Read as text file
with open(path, "r", encoding="utf-8") as f:
txt = f.read()

if not txt:
continue

# Insert delimiters after every 100 words
txt = insert_delimiters_every_n_words(txt, n=100, delimiter=DELIMITER)
chunks.extend(list(split_by_delimiter(txt, DELIMITER)))

return chunks

# SEARCH (phrase-only; uses double-quoted phrases in user question)
QUOTED_PHRASE_RX = re.compile(r'"([^"]+)"')

def extract_quoted_phrases(text: str) -> list[str]:

phrases = [m.group(1).strip() for m in QUOTED_PHRASE_RX.finditer(text)]
seen = set()
ordered = []
for p in phrases:
key = p.lower()
if p and key not in seen:
seen.add(key)
ordered.append(p)
return ordered

def score_chunk_by_phrases(chunk: str, phrases: list[str]) -> int:
c_lower = chunk.lower()
score = 0
for p in phrases:
pl = p.lower()
start = 0
while True:
idx = c_lower.find(pl, start)
if idx == -1:
break
score += 1
start = idx + len(pl)
return score

def find_top_chunks(chunks: list[str], question: str, top_k: int) -> list[str]:
"""
LLM-assisted retrieval:
1) Expand the question into canonical phrases, synonyms, and keywords via LLM.
2) Score chunks with weighted matching.
3) If LLM expansion fails or yields nothing, fall back to phrase-only search based on double quotes.
"""
# 1) LLM expansion
canonical, synonyms, keywords = [], [], []
try:
expansion = llm_expand_query(question)
if expansion:
canonical, synonyms, keywords = build_terms_from_expansion(expansion)
except Exception:
# Swallow errors and fall back gracefully
canonical, synonyms, keywords = [], [], []

results: list[tuple[int, str]] = []

# 2) Use weighted scoring if we have any signals
if canonical or synonyms or keywords:
for c in chunks:
s = score_chunk_weighted(c, canonical, synonyms, keywords)
if s > 0:
results.append((s, c))

if results:
results.sort(key=lambda x: x[0], reverse=True)
return [c for _, c in results[: max(1, top_k)]]

# 3) Fallback to phrase-only search using double-quoted phrases
phrases = extract_quoted_phrases(question)
if not phrases:
return [] # strict fallback behavior if no phrases exist

for c in chunks:
s = score_chunk_by_phrases(c, phrases)
if s > 0:
results.append((s, c))

results.sort(key=lambda x: x[0], reverse=True)
return [c for _, c in results[: max(1, top_k)]]



# PROMPT BUILDING
def build_user_prompt(question: str, chunks: list[str]) -> str:
sb = []
sb.append("QUESTION:")
sb.append(question)
sb.append("")
sb.append("CONTEXT CHUNKS:")
for i, c in enumerate(chunks):
sb.append(f"[Chunk {i+1}]")
sb.append(c)
sb.append("")
sb.append("Answer:")
return "\n".join(sb)

def trim_to_fit(system_prompt: str, user_prompt: str, max_total_chars: int) -> str:
budget_for_user = max_total_chars - len(system_prompt)
if budget_for_user <= 0:
return ""

if len(user_prompt) <= budget_for_user:
return user_prompt

suffix = "\nAnswer:"
if budget_for_user > len(suffix) + 50:
head_len = budget_for_user - len(suffix)
return user_prompt[:head_len] + suffix

return user_prompt[:budget_for_user]

# Save prompt locally for debugging/evaluation
def save_prompt(system_prompt: str, user_prompt: str, top_chunks: list[str], out_dir: str | None = None) -> str:
if out_dir is None:
out_dir = os.path.join(os.path.expanduser("~"), "Desktop", "LLM", "rag_prompts")
os.makedirs(out_dir, exist_ok=True)
meta = {
"timestamp": datetime.utcnow().isoformat() + "Z",
"id": uuid4().hex,
"system_chars": len(system_prompt),
"user_chars": len(user_prompt),
"total_chars": len(system_prompt) + len(user_prompt),
"top_chunks_count": len(top_chunks),
}
payload = {
"meta": meta,
"system_prompt": system_prompt,
"user_prompt": user_prompt,
"top_chunks": top_chunks,
}
fname = f"prompt_{meta['timestamp'].replace(':','').replace('-','')}_{meta['id'][:8]}.json"
path = os.path.join(out_dir, fname)
with open(path, "w", encoding="utf-8") as f:
json.dump(payload, f, ensure_ascii=False, indent=2)
return path


def call_github_models_json(system_prompt: str, user_prompt: str, model: str = SEARCH_MODEL, max_tokens: int = SEARCH_MAX_TOKENS) -> dict:
"""
Calls GitHub Models and attempts to parse a JSON object from the response content.
Returns {} on any failure.
"""
if not GITHUB_PAT or GITHUB_PAT == "PUT_YOUR_GITHUB_PAT_HERE":
return {}

headers = {
"Accept": "application/vnd.github+json",
"Authorization": f"Bearer {GITHUB_PAT}",
"X-GitHub-Api-Version": "2022-11-28",
"Content-Type": "application/json",
}

payload = {
"model": model,
"messages": [
{"role": "system", "content": system_prompt},
{"role": "user", "content": user_prompt},
],
"max_tokens": max_tokens,
"temperature": 0.0,
}

try:
r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
if r.status_code >= 300:
return {}
data = r.json()
text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
if not text:
return {}

# Robust JSON extraction: find the largest {...} block
start = text.find("{")
end = text.rfind("}")
if start == -1 or end == -1 or end <= start:
return {}
blob = text[start:end+1]
return json.loads(blob)
except Exception:
return {}


# --- Phrase-only fallback (double-quoted) utilities ---
QUOTED_PHRASE_RX = re.compile(r'"([^"]+)"')

def extract_quoted_phrases(text: str) -> list[str]:
phrases = [m.group(1).strip() for m in QUOTED_PHRASE_RX.finditer(text)]
seen = set()
ordered = []
for p in phrases:
key = p.lower()
if p and key not in seen:
seen.add(key)
ordered.append(p)
return ordered

def score_chunk_by_phrases(chunk: str, phrases: list[str]) -> int:
c_lower = chunk.lower()
score = 0
for p in phrases:
pl = p.lower()
start = 0
while True:
idx = c_lower.find(pl, start)
if idx == -1:
break
score += 1
start = idx + len(pl)
return score


# --- LLM-assisted query expansion and weighted scoring ---
def llm_expand_query(question: str) -> dict:
"""
Uses the LLM to expand the question into canonical phrases, synonyms, and keywords.
Returns {} if disabled or if the LLM cannot be used.
"""
if not ENABLE_LLM_SEARCH:
return {}

user_prompt = (
"User question:\n"
f"{question}\n\n"
"Return JSON with keys: canonical_phrases, synonyms, keywords."
)
res = call_github_models_json(SEARCH_SYSTEM_PROMPT, user_prompt, model=SEARCH_MODEL, max_tokens=SEARCH_MAX_TOKENS)
# Basic shape validation
if not isinstance(res, dict):
return {}
if not any(k in res for k in ("canonical_phrases", "synonyms", "keywords")):
return {}
return res


def _dedup_ordered(items: list[str]) -> list[str]:
seen = set()
out = []
for it in items:
s = (it or "").strip()
key = s.lower()
if s and key not in seen:
seen.add(key)
out.append(s)
return out


def build_terms_from_expansion(exp: dict) -> tuple[list[str], list[str], list[str]]:
"""
Normalizes and de-duplicates terms from the LLM expansion.
Returns (canonical_phrases, synonyms_flat, keywords)
"""
canonical = _dedup_ordered(list(exp.get("canonical_phrases", []) or []))

# synonyms: dict[canonical -> [aliases...]]
syn_flat = []
syn_map = exp.get("synonyms", {}) or {}
if isinstance(syn_map, dict):
for _, arr in syn_map.items():
if isinstance(arr, list):
syn_flat.extend(arr)
synonyms = _dedup_ordered(syn_flat)

keywords = _dedup_ordered(list(exp.get("keywords", []) or []))
return (canonical, synonyms, keywords)


def _count_occurrences(haystack_lower: str, needle_lower: str) -> int:
"""
Non-overlapping substring occurrence count, case-insensitive (input expected lowercased).
"""
if not needle_lower:
return 0
count = 0
start = 0
while True:
idx = haystack_lower.find(needle_lower, start)
if idx == -1:
break
count += 1
start = idx + len(needle_lower)
return count


def score_chunk_weighted(chunk: str, canonical: list[str], synonyms: list[str], keywords: list[str]) -> int:
"""
Weighted scoring with boundary-aware matches and structural boosts for date/validity.
"""
score = 0
cl = chunk # keep original case for regex; helpers lowercase internally

# Strong boost for explicit Agreement Date phrasing
for p in AGREEMENT_DATE_PHRASES:
score += 4 * count_word_boundary_occurrences(cl, p)

# Canonical / synonyms (multi-word allowed)
for p in canonical:
score += 3 * count_word_boundary_occurrences(cl, p)
for s in synonyms:
score += 2 * count_word_boundary_occurrences(cl, s)

# Keywords (boundary-aware for single tokens)
for k in keywords:
score += 1 * count_word_boundary_occurrences(cl, k)

# Structural boosts
if has_any(DATE_RX_LIST, cl):
score += 5 # contains a date-like string
if has_any(VALIDITY_RX_LIST, cl):
score += 4 # contains validity/term language

# If chunk mentions both "agreement" and "date" anywhere, nudge up
if re.search(r"\bagreement\b", cl, re.I) and re.search(r"\bdate\b", cl, re.I):
score += 3

return score


# LLM CALL (HTTP POST)
def call_github_models(system_prompt: str, user_prompt: str) -> str:
if not GITHUB_PAT or GITHUB_PAT == "PUT_YOUR_GITHUB_PAT_HERE":
raise RuntimeError("GitHub PAT not configured. Set GITHUB_PAT env var or edit GITHUB_PAT in code.")

headers = {
"Accept": "application/vnd.github+json",
"Authorization": f"Bearer {GITHUB_PAT}",
"X-GitHub-Api-Version": "2022-11-28",
"Content-Type": "application/json",
}

payload = {
"model": MODEL,
"messages": [
{"role": "system", "content": system_prompt},
{"role": "user", "content": user_prompt},
],
"max_tokens": MAX_COMPLETION_TOKENS, # output tokens cap
}

r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
if r.status_code >= 300:
raise RuntimeError(f"GitHub Models error {r.status_code}: {r.text}")

data = r.json()
return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip() or "(empty response)"

# Response save
def append_response_log(response: str, out_path: str = RESPONSES_LOG_FILE, doc_name: str = "") -> str:
"""
Appends the LLM response to a CSV file with columns:
run_number, timestamp, name, result
Returns the file path.
"""
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# Determine next run number by scanning existing file
run_num = 0
file_exists = os.path.exists(out_path)

if file_exists:
with open(out_path, "r", encoding="utf-8", newline="") as f:
reader = csv.DictReader(f)
for row in reader:
if row and "run_number" in row:
try:
n = int(row["run_number"])
run_num = max(run_num, n)
except (ValueError, TypeError):
pass

run_num += 1
timestamp = datetime.now().isoformat()

# Write to CSV
with open(out_path, "a", encoding="utf-8", newline="") as f:
writer = csv.DictWriter(f, fieldnames=["run_number", "timestamp", "name", "result"])

# Write header if file is new
if not file_exists:
writer.writeheader()

writer.writerow({
"run_number": run_num,
"timestamp": timestamp,
"name": doc_name,
"result": response
})

return out_path



# MAIN with batch processing
def process_batch(files: list[str], user_question: str):
"""
Process a batch of files with the same user question.
"""
if not files:
print("[warning] No files to process in batch")
return

print(f"\n[batch] Processing {len(files)} file(s): {[os.path.basename(f) for f in files]}")

chunks = load_chunks(files)
if not chunks:
print("[error] No chunks loaded from files")
return

top = find_top_chunks(chunks, user_question, TOP_K)
trimmed = [(c[:MAX_CHUNK_CHARS] if len(c) > MAX_CHUNK_CHARS else c) for c in top]
user_prompt = build_user_prompt(user_question, trimmed)
user_prompt = trim_to_fit(SYSTEM_PROMPT, user_prompt, MAX_INPUT_CHARS)

print(f"[prompt] systemChars={len(SYSTEM_PROMPT)} userChars={len(user_prompt)} totalChars={len(SYSTEM_PROMPT)+len(user_prompt)}")

# Save prompt for evaluation/debugging
saved_prompt_path = save_prompt(SYSTEM_PROMPT, user_prompt, trimmed)
print(f"[prompt saved] {saved_prompt_path}")

answer = call_github_models(SYSTEM_PROMPT, user_prompt)
print(f"[answer] {answer}")

# Log result for each file in the batch
doc_names = ", ".join([os.path.basename(f) for f in files])
log_path = append_response_log(answer, RESPONSES_LOG_FILE, doc_names)
print(f"[response saved] {log_path}")


def main():
# USER PROMPT LINE (edit this)
user_question = 'Find the Date of Agreement and the term/validity period. Determine if it has expired as of January 30, 2026. Reply only with "Expired" or "Valid".'
if len(user_question) > MAX_QUESTION_CHARS:
user_question = user_question[:MAX_QUESTION_CHARS]

# Get all PDFs from folder
pdf_files = get_pdf_files_from_folder(PDF_FOLDER)

if not pdf_files:
print(f"[info] No PDFs found in {PDF_FOLDER}")
print("[info] Falling back to DOC_FILES configuration")
process_batch(DOC_FILES, user_question)
return

print(f"[info] Found {len(pdf_files)} PDF(s) in {PDF_FOLDER}")
print(f"[rate-limit] Processing at {PDFS_PER_MINUTE} PDFs per 60 seconds ({DELAY_BETWEEN_PDFS:.1f}s between each)")

# Process PDFs one at a time with rate limiting
for idx, pdf_file in enumerate(pdf_files, 1):
batch = [pdf_file]
print(f"\n[{idx}/{len(pdf_files)}] Processing PDF: {os.path.basename(pdf_file)}")
process_batch(batch, user_question)

# Wait before next PDF (except for the last one)
if idx < len(pdf_files):
print(f"[rate-limit] Waiting {DELAY_BETWEEN_PDFS:.1f} seconds before next PDF...")
time.sleep(DELAY_BETWEEN_PDFS)


if __name__ == "__main__":
main()
