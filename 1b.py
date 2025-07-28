import fitz
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# Regexes for heading detection & filtering
RE_NUMBERING = re.compile(r"^(\d+(\.\d+)+)\s+(.*)")
RE_BULLET = re.compile(r"[-•‣◦\*\u2022]+")
RE_NONWORD = re.compile(r"[\W_]+")
RE_ALLNUMS = re.compile(r"[\d\s]+")

def clean_text(text: str) -> str:
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("—", "-")
    return " ".join(text.split())

def extract_page_lines(page_num_page_tuple):
    page_num, page = page_num_page_tuple
    page_dict = page.get_text("dict")
    page_lines = []
    for b in page_dict.get("blocks", []):
        if b.get("type") == 0:
            for l in b.get("lines", []):
                line_text = clean_text(
                    "".join(s.get("text", "") for s in l.get("spans", []))
                )
                if line_text and l.get("spans"):
                    s = l["spans"][0]
                    page_lines.append(
                        {
                            "text": line_text,
                            "font_size": round(s["size"]),
                            "is_bold": "bold" in s["font"].lower(),
                            "page_num": page_num,
                            "bbox": l["bbox"],
                        }
                    )
    return page_lines

def get_line_blocks_parallel(doc):
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_page_lines, enumerate(doc, start=1))
    all_lines = []
    for lines in results:
        all_lines.extend(lines)
    return all_lines

def remove_headers_and_footers(lines, doc):
    if not lines:
        return []
    page_height = doc[0].rect.height
    headers = [l for l in lines if l["bbox"][1] < page_height * 0.1]
    footers = [l for l in lines if l["bbox"][3] > page_height * 0.9]
    header_texts = Counter(l["text"] for l in headers)
    footer_texts = Counter(l["text"] for l in footers)
    headers_to_remove = {text for text, cnt in header_texts.items() if cnt > 1}
    footers_to_remove = {text for text, cnt in footer_texts.items() if cnt > 1}
    return [
        l for l in lines if (l["text"] not in headers_to_remove) and (l["text"] not in footers_to_remove)
    ]

def is_japanese(text):
    return any("\u3040" <= c <= "\u309F" or "\u30A0" <= c <= "\u30FF" or "\u4E00" <= c <= "\u9FFF" for c in text)

def is_devanagari(text):
    return any("\u0900" <= c <= "\u097F" for c in text)

def is_false_positive(text):
    text = text.strip()
    if not (is_japanese(text) or is_devanagari(text)):
        if len(text) < 3:
            return True
    if RE_BULLET.fullmatch(text):
        return True
    if RE_NONWORD.fullmatch(text):
        return True
    if RE_ALLNUMS.fullmatch(text):
        return True
    if text.replace(" ", "") == "फामाॉहाथदामाॉहाथ":
        return True
    return False

def merge_fragmented_headings(outline):
    if not outline:
        return []
    merged = []
    i = 0
    while i < len(outline):
        current = outline[i]
        if is_false_positive(current["text"]):
            i += 1
            continue
        is_num = re.fullmatch(r"(\d+(\.\d+)+)\.?", current["text"].strip())
        if i + 1 < len(outline):
            nxt = outline[i + 1]
            nxt_is_num = re.fullmatch(r"(\d+(\.\d+)+)\.?", nxt["text"].strip())
            if (
                is_num
                and not nxt_is_num
                and current["page"] == nxt["page"]
                and not is_false_positive(nxt["text"])
            ):
                level = "H" + str(current["text"].count(".") + 1)
                merged.append(
                    {
                        "level": level,
                        "text": f"{current['text']} {nxt['text']}",
                        "page": current["page"],
                    }
                )
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged

def extract_title_and_outline(doc):
    # Robust outline extraction following the Round 1A logic
    lines = get_line_blocks_parallel(doc)
    if not lines:
        return "", []

    doc_npages = doc.page_count
    if len(lines) < 35:
        # Small doc heuristic
        first_page_lines = [l for l in lines if l["page_num"] == 1]
        if not first_page_lines:
            return "", []
        max_font_line = max(first_page_lines, key=lambda x: x["font_size"])
        title = max_font_line["text"]
        outline = []
        return title, outline

    first_page_lines = [l for l in lines if l["page_num"] == 1]
    page_height = doc[0].rect.height
    title, title_lines = extract_multiline_title(first_page_lines, page_height)

    lines = [l for l in lines if l["text"] not in title_lines]
    doc = remove_headers_and_footers(lines, doc)

    # Filter out table of contents pages
    toc_pages = {l["page_num"] for l in lines if "Table of Contents" in l["text"]}
    lines = [l for l in lines if l["page_num"] not in toc_pages]

    candidates = [
        l
        for l in lines
        if len(l["text"].split()) < 25 and not l["text"].strip().endswith((".", ":"))
    ]

    if not candidates:
        return title, []

    font_sizes = [l["font_size"] for l in candidates]
    most_common_font_size = Counter(font_sizes).most_common(1)[0][0]

    preliminary_outline = []
    for line in candidates:
        level = None
        if RE_NUMBERING.match(line["text"]) and line["font_size"] >= most_common_font_size:
            level_num = line["text"].count(".") + 1
            if level_num <= 3:
                level = "H" + str(level_num)
        elif line["is_bold"] and line["font_size"] > most_common_font_size:
            if not line["text"].isupper() or len(line["text"].split()) > 2:
                level = "H1"
        elif line["font_size"] > most_common_font_size + 1 and not line["is_bold"]:
            if len(line["text"].split()) < 8:
                level = "H1"
        if level:
            preliminary_outline.append(
                {"level": level, "text": line["text"], "page": line["page_num"]}
            )

    outline = merge_fragmented_headings(preliminary_outline)
    return title, outline

def extract_multiline_title(lines, page_height):
    # Extract multiline title heuristic
    if not lines:
        return "", []
    max_font = max(l["font_size"] for l in lines)
    threshold = max_font * 0.85
    title_lines = [l for l in lines if l["font_size"] >= threshold]
    title_lines = sorted(title_lines, key=lambda x: x["bbox"][1])
    title = " ".join([l["text"] for l in title_lines])
    return title.strip(), [l["text"] for l in title_lines]

def extract_snippet(doc, page_num, title, outline, idx):
    """Extract text under heading, may span multiple pages until next heading."""
    snippet_lines = []

    def collect_text(pg, start_after):
        try:
            page = doc[pg - 1]
        except:
            return []
        blocks = page.get_text("blocks")
        collected = []
        start_collect = False
        for b in sorted(blocks, key=lambda x: x[1]):
            txt = b[4].strip()
            if not start_collect:
                if start_after in txt:
                    start_collect = True
                else:
                    continue
            else:
                if any(o["text"] in txt for o in outline if o["page"] == pg and o["text"] != title):
                    break
                if txt != "":
                    collected.append(txt)
        return collected

    snippet_lines.extend(collect_text(page_num, title))

    # If next heading exists on a later page, collect those pages fully
    next_pg = outline[idx + 1]["page"] if idx + 1 < len(outline) else None
    if next_pg is not None and next_pg > page_num:
        for p in range(page_num + 1, next_pg):
            try:
                snippet_lines.append(doc[p - 1].get_text())
            except:
                pass

    snippet = "\n".join(snippet_lines)
    if snippet.strip() == "":
        # fallback: get whole page text minus the title
        try:
            page_text = doc[page_num - 1].get_text().replace(title, "")
            snippet = page_text.strip()
        except:
            snippet = ""
    return snippet[:1500]

def mmr(doc_embs, query_emb, lambda_val=0.7, top_k=5):
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    query_emb /= np.linalg.norm(query_emb)
    similarities = (doc_embs @ query_emb.T).flatten()
    selected = []
    candidates = set(range(len(doc_embs)))

    while len(selected) < top_k and candidates:
        if not selected:
            next_idx = max(candidates, key=lambda x: similarities[x])
            selected.append(next_idx)
            candidates.remove(next_idx)
        else:
            mmr_scores = []
            for c in candidates:
                relevance = similarities[c]
                diversity = max(
                    cosine_similarity(doc_embs[c].reshape(1, -1), doc_embs[selected]).flatten()
                )
                mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
                mmr_scores.append((mmr_score, c))
            mmr_scores.sort(reverse=True)
            best_score, best_idx = mmr_scores[0]
            if best_score <= 0:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)
    return selected

def clean_long_title(title: str) -> str:
    MAX_WORDS = 15
    title = title.strip()
    words = title.split()
    if len(words) > MAX_WORDS:
        title = " ".join(words[:MAX_WORDS]) + "..."
    return title.rstrip(",.;:")

def main():
    parser = argparse.ArgumentParser(description="Run Round 1B Document Intelligence Pipeline")
    parser.add_argument(
        "--input_json", type=str, required=True, help="Path to input JSON with persona, documents etc."
    )
    parser.add_argument(
        "--pdf_folder", type=str, default="pdf_folder", help="Folder containing input PDFs"
    )
    parser.add_argument(
        "--output_folder", type=str, default="1b_output", help="Folder to write output JSON"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top sections to output"
    )
    parser.add_argument(
        "--use_mmr", action="store_true", help="Use Maximum Marginal Relevance for diversity"
    )
    args = parser.parse_args()

    # Load input JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    persona = input_data.get("persona", {}).get("role", "")
    job = input_data.get("job_to_be_done", {}).get("task", "")
    documents = input_data.get("documents", [])

    files_to_process = [doc["filename"] for doc in documents]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Load model once
    model = SentenceTransformer('all-MiniLM-L6-v2')

    metadata = {
        "input_documents": files_to_process,
        "persona": persona,
        "job": job,
        "processing_timestamp": str(datetime.datetime.now()),
    }

    candidate_sections = []

    query_text = f"{persona} {job}".strip()
    query_emb = model.encode([query_text])

    for doc_fn in files_to_process:
        pdf_path = os.path.join(args.pdf_folder, doc_fn)
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file {doc_fn} not found in {args.pdf_folder}, skipping.")
            continue

        doc = fitz.open(pdf_path)
        title, outline = extract_title_and_outline(doc)

        doc_sections = []
        if title:
            snippet = extract_snippet(doc, 1, title, outline, -1)
            doc_sections.append({"document": doc_fn, "page_num": 1, "section_title": title, "content": snippet})

        for idx, heading in enumerate(outline):
            page_no = heading["page"]
            sec_title = heading["text"]
            snippet = extract_snippet(doc, page_no, sec_title, outline, idx)
            doc_sections.append(
                {
                    "document": doc_fn,
                    "page_num": page_no,
                    "section_title": sec_title,
                    "content": snippet,
                }
            )

        if not doc_sections:
            continue

        texts = [s["section_title"] + " " + s["content"] for s in doc_sections]
        embeddings = model.encode(texts)
        sims = cosine_similarity(embeddings, query_emb).flatten()

        top_indices = sims.argsort()[::-1][: args.top_k]
        for idx in top_indices:
            section = doc_sections[idx]
            section["similarity"] = float(sims[idx])
            candidate_sections.append(section)

    if not candidate_sections:
        print("No document sections extracted.")
        return

    combined_texts = [c["section_title"] + " " + c["content"] for c in candidate_sections]
    combined_embeds = model.encode(combined_texts)

    if args.use_mmr:
        selected_indices = mmr(combined_embeds, query_emb, lambda_val=0.7, top_k=args.top_k)
    else:
        all_sims = cosine_similarity(combined_embeds, query_emb).flatten()
        selected_indices = all_sims.argsort()[::-1][: args.top_k]

    selected_sections = [candidate_sections[i] for i in selected_indices]

    for rank, sec in enumerate(selected_sections, start=1):
        sec["importance_rank"] = rank

    subsections = []
    for sec in selected_sections:
        snippet = sec.get("content", "")
        if snippet:
            sents = sent_tokenize(snippet)
            if sents:
                sent_embeds = model.encode(sents)
                s_scores = cosine_similarity(sent_embeds, query_emb).flatten()
                top_sent_idxs = s_scores.argsort()[::-1][:3]
                refined_text = " ".join(sents[i] for i in top_sent_idxs).replace("\n", " ").strip()
            else:
                refined_text = ""
        else:
            refined_text = ""
        subsections.append(
            {"document": sec["document"], "refined_text": refined_text, "page_number": sec["page_num"]}
        )

    final_sections = [
        {
            "document": sec["document"],
            "page_number": sec["page_num"],
            "section_title": clean_long_title(sec["section_title"]),
            "importance_rank": sec["importance_rank"],
        }
        for sec in selected_sections
    ]

    output_data = {
        "metadata": metadata,
        "extracted_sections": final_sections,
        "subsection_analysis": subsections,
    }

    out_file_path = os.path.join(args.output_folder, "round1b_output.json")
    with open(out_file_path, "w", encoding="utf-8") as f_out:
        json.dump(output_data, f_out, indent=2)

    print(f"Processed {len(files_to_process)} documents.")
    print(f"Output written to {out_file_path}")


def clean_long_title(title: str) -> str:
    MAX_WORDS = 15
    title = title.strip()
    words = title.split()
    if len(words) > MAX_WORDS:
        title = " ".join(words[:MAX_WORDS]) + "..."
    return title.rstrip(",.;:)")


if __name__ == "__main__":
    main()