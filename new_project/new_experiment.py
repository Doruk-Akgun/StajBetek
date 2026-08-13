import json
import os
import re
import time
import chromadb
import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from pdf_reading_order import extract_pdf_text
from pdf_extraction import extract_paint_properties


# ---------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

DATASET_PATH = "dataset"
PDF_FILES =["TDS_AQUALUX.pdf",
            "TDS_EXXENMAT.pdf",
            "TDS_MOMENTOPLASTIX.pdf",
            "TDS_WOODMAXX_WOODSTAIN.pdf",
            "TDS_MOMENTOSILAN.pdf"]

PDF_PATHS = [os.path.join(DATASET_PATH, file) for file in PDF_FILES]

PRODUCT_NAMES = [
    "AquaLux",
    "Exxen Mat",
    "Momento Plastix",
    "WoodMaXX Wood Stain Dekoratif Ahşap Verniği",
    "Momento Silan",
]

CHROMA_DB_PATH = "./paint_db"
COLLECTION_NAME = "test7"

# Structured properties extracted per document during ingestion, keyed by
# doc_id (filled in by ingest_pdf via extract_paint_properties(pages)).
PAINT_PROPERTIES = {}

PARENT_CHUNK_SIZE = 1024
PARENT_OVERLAP = 128
CHILD_CHUNK_SIZE = 256
CHILD_OVERLAP = 32

VECTOR_TOP_K = 15 # Not being used rn
BM25_TOP_K = 15 # Not being used rn
FINAL_TOP_K = 5
RRF_K = 60  # standard RRF damping constant

# Confidence gating for similarity-search results (semantic / hybrid). 
# Confidence is a 0..1 score derived from the retrieval
# signals a record actually carries (see compute_confidence()).
#   score <  CONFIDENCE_DROP_THRESHOLD -> discarded, never shown to the LLM
#   score <  CONFIDENCE_LOW_THRESHOLD  -> kept, but tagged "low confidence"
#   score >= CONFIDENCE_LOW_THRESHOLD  -> kept, shown normally
# Exact metadata matches (metadata_question/list/comparison) never go
# through this gate -- they're filter lookups, not nearest-neighbor
# guesses, so they're always confidence 1.0.
CONFIDENCE_DROP_THRESHOLD = 0.35
CONFIDENCE_LOW_THRESHOLD = 0.55

# Widened candidate pool for "default_qa"
DEFAULT_QA_CANDIDATE_K = 40


SEARCH_MODE = "hybrid_parent_child"




# ---------------------------------------------------------
# 1a LM Studio embedding function
# ---------------------------------------------------------
def get_embedding(text):
    url = f"{LM_STUDIO_BASE_URL}/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"input": text, "model": "local"}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise Exception(f"LM Studio Error: {response.status_code} - {response.text}")
    return response.json()["data"][0]["embedding"]


# ---------------------------------------------------------
# 1b LM Studio chat/completion function
# ---------------------------------------------------------
def ask_llm(system_prompt, user_prompt):
    url = f"{LM_STUDIO_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise Exception(f"LM Studio Error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"]




# ---------------------------------------------------------
# 2a. Splitting for chunking
# ---------------------------------------------------------
def split_into_pieces(text, chunk_size, overlap):
    pieces = []
    i = 0
    while i < len(text):
        piece = text[i : i + chunk_size].strip()
        if piece:
            pieces.append(piece)
        i += chunk_size - overlap
    return pieces

# ---------------------------------------------------------
# 2b. Parent-Child  chunking
# ---------------------------------------------------------
def chunk_text_parent_child(pages, doc_id):
    parent_chunks = []
    child_chunks = []

    parent_counter = 0
    for page in pages:
        parent_pieces = split_into_pieces(page["text"], PARENT_CHUNK_SIZE, PARENT_OVERLAP)
        for parent_text in parent_pieces:
            parent_id = f"{doc_id}_parent_{parent_counter}"
            parent_counter += 1
            parent_chunks.append(
                {"parent_id": parent_id, "text": parent_text, "page_number": page["page_number"]}
            )

            child_pieces = split_into_pieces(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)
            for i, text in enumerate(child_pieces):
                child_chunks.append(
                    {"parent_id": parent_id, "text": text,
                    "page_number": page["page_number"],
                    "chunk_index_in_page": i, 
                    "chunk_length": len(text)}
                )

    return parent_chunks, child_chunks


# ---------------------------------------------------------
# 3a. Properties for ingestion
# ---------------------------------------------------------
def properties_to_metadata(properties):
    meta = {"properties_json": json.dumps(properties, ensure_ascii=False)}
    for key, value in properties.items():
        if value is None or value == [] or value == "":
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        elif not isinstance(value, (str, int, float, bool)):
            value = str(value)
        meta[f"prop_{key}"] = value
    return meta


# ---------------------------------------------------------
# 3b. Ingestion
# ---------------------------------------------------------
def ingest_pdf(pdf_path, collection, parent_collection, pdf_file):
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"   Reading '{pdf_path}'...")
    pages = extract_pdf_text(pdf_path)
    print(f"   Extracted text from {len(pages)} pages.")

    # extract_paint_properties takes the `pages` list itself (the output
    # of extract_pdf_text above), never the raw pdf_path.
    properties = extract_paint_properties(pages)
    PAINT_PROPERTIES[doc_id] = properties
    print(f"   Extracted paint properties for '{doc_id}':")

    # Flatten once per document; every parent AND child chunk below gets
    # this same dict merged into its metadata, so the structured info
    # travels with each cluster of chunks for this product and is reachable
    # from parent chunks too (not just stashed in a separate record).
    properties_meta = properties_to_metadata(properties)

    parent_chunks, child_chunks = chunk_text_parent_child(pages, doc_id)
    print(f"   Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks.")

    for parent in parent_chunks:
        parent_collection.add(
            embeddings=[[0.0]],
            documents=[parent["text"]],
            metadatas=[{
                "source": pdf_file,
                "page_number": parent["page_number"],
                **properties_meta,
            }],
            ids=[parent["parent_id"]],
        )

    for i, child in enumerate(child_chunks):
        vector = get_embedding(child["text"])
        child_id = f"{doc_id}_child_{i}"
        collection.add(
            embeddings=[vector],
            documents=[child["text"]],
            metadatas=[{
                "source": pdf_file,
                "page_number": child["page_number"],
                "child_id": child_id,
                "parent_id": child["parent_id"],
                "chunk_index_in_page": child["chunk_index_in_page"],
                "chunk_length": child["chunk_length"],
                **properties_meta,
            }],
            ids=[child_id],
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(child_chunks):
            print(f"   [{i + 1}/{len(child_chunks)}] child chunks processed.")

    print(f"   Finished ingesting '{pdf_path}': {len(parent_chunks)} parents, {len(child_chunks)} children.\n")


# ---------------------------------------------------------
# 4. Build a BM25 index over every child chunk currently in Chroma
# ---------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_bm25_index(collection):
    all_children = collection.get(include=["documents", "metadatas"])
    ids = all_children["ids"]
    documents = all_children["documents"]
    metadatas = all_children["metadatas"]

    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }


# ---------------------------------------------------------
# 5a. Reciprocal Rank Fusion
# ---------------------------------------------------------
def reciprocal_rank_fusion(ranked_id_lists, k=RRF_K):
    scores = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in fused]


# ---------------------------------------------------------------------------
# 5B. Confidence scoring -- turns whatever similarity signal a record
#     already carries into one comparable 0..1 number, then gates on it.
# ---------------------------------------------------------------------------

def _dense_similarity(distance):
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - distance))


def _normalize_sparse_scores(records):
    scored = [(id(r), r["sparse_score"]) for r in records if r.get("sparse_score") is not None]
    if not scored:
        return {}
    values = [v for _, v in scored]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {rid: 1.0 for rid, _ in scored}
    return {rid: (v - lo) / (hi - lo) for rid, v in scored}


def compute_confidence(records):
    """Attaches `confidence` (0..1 float) and `confidence_label`
    ("high"/"low"/"drop") to every record, in place. Uses whichever
    signal that record actually has:

    - "distance" (flat semantic_search records) -> dense similarity alone.
    - "dense_distance" (+ optional "sparse_score", hybrid_search records)
      -> a weighted blend, dense weighted higher since it's on an
      absolute query-independent scale while BM25 is only meaningful
      relative to this batch.
    - neither key present (metadata_question/list/comparison records,
      which were fetched by exact filter via collection.get(), never
      ranked by similarity) -> confidence = 1.0.

    Returns `records` (same list, mutated) so callers can chain this
    into a filter step without an extra variable.
    """
    sparse_norm = _normalize_sparse_scores(records)
    for r in records:
        dense_dist = r.get("distance", r.get("dense_distance"))
        dense_sim = _dense_similarity(dense_dist)
        sparse_sim = sparse_norm.get(id(r))

        if dense_sim is None and sparse_sim is None:
            confidence = 1.0  # exact metadata match, not a similarity guess
        elif dense_sim is not None and sparse_sim is not None:
            confidence = 0.7 * dense_sim + 0.3 * sparse_sim
        elif dense_sim is not None:
            confidence = dense_sim
        else:
            confidence = sparse_sim

        r["confidence"] = round(confidence, 4)
        if confidence < CONFIDENCE_DROP_THRESHOLD:
            r["confidence_label"] = "drop"
        elif confidence < CONFIDENCE_LOW_THRESHOLD:
            r["confidence_label"] = "low"
        else:
            r["confidence_label"] = "high"
    return records


def apply_confidence_thresholds(records):
    compute_confidence(records)
    kept = [r for r in records if r["confidence_label"] != "drop"]
    return kept


def _confidence_tag(record):
    if record.get("confidence_label") == "low":
        return "[DÜŞÜK GÜVEN] "
    return ""


# ---------------------------------------------------------
# 6. Deterministic query analysis.
#     Filter/intent extraction is now deterministic-first (regex/keyword
#     rules over the question), reusing the same _deterministic_filters /
#     _PROPERTY_KEYWORDS helpers the old LLM-assisted version already
#     had. An LLM call only happens for the narrow slice of questions
#     the rules genuinely can't disambiguate (see analyze_query below),
#     and even then it returns just {intent, requested_property} -- no
#     filters, no query rewriting, no field documentation in the prompt.
# ---------------------------------------------------------


_PROPERTY_DISPLAY_LABELS = {
    "prop_su_bazlı": "Su bazlı",
    "prop_voc_uyumlu": "VOC uyumlu",
    "prop_kullanım_alanı": "Kullanım alanı",
    "prop_doku": "Doku",
    "prop_sarfiyat_min": "Sarfiyat",
    "prop_sarfiyat_max": "Sarfiyat",
    "prop_depolama_süresi": "Depolama süresi",
    "prop_ambalaj_boyutları": "Ambalaj boyutları",
    "prop_dokunma_kuruması_min": "Dokunma kuruması",
    "prop_dokunma_kuruması_max": "Dokunma kuruması",
    "prop_katlar_arası_bekleme_min": "Katlar arası bekleme süresi",
    "prop_katlar_arası_bekleme_max": "Katlar arası bekleme süresi",
    "prop_son_kuruma": "Son kuruma",
    "prop_inceltme_havasız_püskürtme": "İnceltme (havasız püskürtme)",
    "prop_inceltme_fırça_rulo": "İnceltme (fırça/rulo)",
}

# Cues that mark a question as asking FOR a single field's value ("what
# is X" / "is it X") rather than asking to FILTER/LIST products that
# already have a stated value for X. Word-boundary regexes so "mi" as a
# question particle doesn't fire on unrelated words containing "mi".
_LOOKUP_CUE_RE = re.compile(
    r"\bnedir\b|\bne\s*kadar\b|\bkaç\b|\bmıdır\b|\bmidir\b|\bmudur\b|\bmüdür\b"
    r"|\bmı\b|\bmi\b|\bmu\b|\bmü\b", re.IGNORECASE)

# Cues that mark a question as asking for a LIST of matching products
# ("which ones", "show me", "list") rather than one field's value.
_LIST_CUE_RE = re.compile(
    r"\bhangi(leri|si)?\b|\blistele\b|\bg[öo]ster\b|\bnelerdir\b|\bneler\b",
    re.IGNORECASE)

# Cues that mark a question as an explicit product-vs-product comparison.
_COMPARISON_CUE_RE = re.compile(
    r"arasındaki|arasında|karşılaştır|kıyasla|\bvs\.?\b|\bcompare\b|\bfark[ıi]?\b",
    re.IGNORECASE)


def normalize_product_name(text):
    """Lowercases and strips whitespace/hyphens/underscores"""
    return re.sub(r"[\s\-_]+", "", text.lower())


def _product_strip_pattern(name):
    """Whitespace/hyphen/underscore-tolerant regex for a single product
    name, used to remove that name from a question's text before running
    property-keyword detection (see _detect_property_keyword /
    _deterministic_filters). A plain str.replace(name.lower(), " ") only
    strips an exact-spacing match, so a spacing variant like 'wood-maxx'
    would survive stripping and could still trigger a false property
    match the same way the unstripped literal name could."""
    parts = re.split(r"[\s\-_]+", name.strip())
    return re.compile(r"[\s\-_]*".join(re.escape(p) for p in parts), re.IGNORECASE)


_PRODUCT_STRIP_PATTERNS = {name: _product_strip_pattern(name) for name in PRODUCT_NAMES}


def _detect_products(user_question):
    q_norm = normalize_product_name(user_question)
    return [(fname, name) for fname, name in zip(PDF_FILES, PRODUCT_NAMES)
            if normalize_product_name(name) in q_norm]


_DRYING_STAGE_FIELDS = (
    "prop_dokunma_kuruması_min",
    "prop_katlar_arası_bekleme_min",
    "prop_son_kuruma",
)

# Generic "does it dry / how long to dry" phrasing that doesn't name a specific stage (dokunma/katlar arası/son) 
# "kaç saatte kurur", "ne kadar sürede kurur", "kuruma süresi nedir", "kuruyor mu".
_GENERIC_DRYING_RE = re.compile(r"\bkuru(r|ma|masi|masına|masının|yor)?\b", re.IGNORECASE)


_THINNING_FIELDS = (
    "prop_inceltme_havasız_püskürtme",
    "prop_inceltme_fırça_rulo",
)
# Generic thinning phrasing with no application method named --
# "ne kadar inceltilmeli", "inceltilmeli mi", "inceltme oranı".
_GENERIC_THINNING_RE = re.compile(r"\bincelt\w*\b", re.IGNORECASE)

def _detect_property_keywords(user_question):
    """Every prop_ field (in _PROPERTY_KEYWORDS order, deduplicated) whose
    keyword hint appears in the question -- e.g. "sarfiyatı ve depolama
    süresi nedir?" detects BOTH prop_sarfiyat_min/max and
    prop_depolama_süresi, not just the first hit. Reuses the exact same
    keyword table _requested_property_is_plausible already used to
    sanity-check the LLM -- now it's the primary detector, not just a
    validator.

    Detected product names are stripped out of the text FIRST. A
    product's own name can accidentally contain a property word (e.g.
    "Exxen Mat" contains "mat", which would otherwise be misread as a
    doku/texture question even when the question is actually asking
    about something else entirely, e.g. "Exxen mat litre başına kaç
    metrekare boyanabilir" is a sarfiyat question, not a doku one)."""
    q = user_question.lower()
    for _fname, name in _detect_products(user_question):
        q = _PRODUCT_STRIP_PATTERNS[name].sub(" ", q)
    found = []
    for field, keywords in _PROPERTY_KEYWORDS.items():
        if any(_keyword_matches(kw, q) for kw in keywords) and field not in found:
            found.append(field)

    # A bare drying question ("aqualux kaç saatte kurur?") doesn't name
    # a specific stage, so none of the three specific keyword lists
    # above ("dokunma kuru", "katlar arası", "son kuruma") match anything
    if not any(f in found for f in _DRYING_STAGE_FIELDS) and _GENERIC_DRYING_RE.search(q):
        found.extend(_DRYING_STAGE_FIELDS)
    
    if not any(f in found for f in _THINNING_FIELDS) and _GENERIC_THINNING_RE.search(q):
        found.extend(_THINNING_FIELDS)

    # Range-sibling fields (e.g. prop_sarfiyat_min / prop_sarfiyat_max)
    # share one keyword list, so a mention of "sarfiyat" matches both --
    # but _lookup_single_property_value already merges both bounds into
    # ONE "min-max unit" answer regardless of which sibling is asked
    # for, so keeping both here would just print the identical merged
    # range twice under duplicate labels. Collapse each min/max pair
    # down to its first-seen sibling.
    deduped = []
    seen_roots = set()
    for field in found:
        root = field[:-4] if field.endswith(("_min", "_max")) else field
        if root in seen_roots:
            continue
        seen_roots.add(root)
        deduped.append(field)
    return deduped


def _deterministic_filters(user_question):
    q = user_question.lower()
    for _fname, name in _detect_products(user_question):
        q = _PRODUCT_STRIP_PATTERNS[name].sub(" ", q)

    filters = {}

    if re.search(r"su\s*bazl[ıi]", q):
        filters["prop_su_bazlı"] = True

    if re.search(r"voc\s*uyumlu", q):
        filters["prop_voc_uyumlu"] = True

    kullanım_match = re.search(r"(iç\s*cephe|dış\s*cephe)", q)
    if kullanım_match:
        filters["prop_kullanım_alanı"] = re.sub(r"\s+", " ", kullanım_match.group(1)).strip()

    doku_match = re.search(
        r"(lüks\s*parlak|yarı\s*parlak|tam\s*parlak|parlak|yarı\s*mat|tam\s*mat|mat)", q)
    if doku_match:
        filters["prop_doku"] = doku_match.group(1).strip()

    return filters


_DOKU_KEYWORD_RE = re.compile(r"doku(?!nma)")

_PROPERTY_KEYWORDS = {
    "prop_su_bazlı": ["su bazl", "su tabanl"],
    "prop_voc_uyumlu": ["voc"],
    "prop_kullanım_alanı": ["kullanım alan", "iç cephe", "dış cephe", "nerede kullan"],
    "prop_doku": [_DOKU_KEYWORD_RE, "mat", "parlak"],
    "prop_sarfiyat_min": ["sarfiyat", "m2", "m²", "metrekare"],
    "prop_sarfiyat_max": ["sarfiyat", "m2", "m²", "metrekare"],
    "prop_depolama_süresi": ["depolama", "raf ömrü", "saklama"],
    "prop_ambalaj_boyutları": ["ambalaj", "litre", "paket"],
    "prop_dokunma_kuruması_min": ["dokunma kuru", "dokunma kurum"],
    "prop_dokunma_kuruması_max": ["dokunma kuru", "dokunma kurum"],
    "prop_katlar_arası_bekleme_min": ["katlar arası", "kat arası", "katlar arasında"],
    "prop_katlar_arası_bekleme_max": ["katlar arası", "kat arası", "katlar arasında"],
    "prop_son_kuruma": ["son kuruma", "tam kuruma", "nihai kuruma"],
    "prop_inceltme_havasız_püskürtme": ["havasız", "airless", "püskürtme inceltme", "inceltme havasız"],
    "prop_inceltme_fırça_rulo": ["fırça", "rulo", "inceltme fırça", "inceltme rulo"],
}


def _keyword_matches(keyword, q):
    """A keyword entry is either a plain substring (most entries -- these
    are deliberately truncated word-stems like 'su bazl' so they also
    catch suffix variants like 'su bazlıdır') or a compiled regex (used
    when a plain substring would collide with an unrelated word, e.g.
    'doku' would otherwise also match inside 'dokunma')."""
    if hasattr(keyword, "search"):
        return keyword.search(q) is not None
    return keyword in q


def _requested_property_is_plausible(requested_property, user_question):
    """False for a field name we don't even recognize, or one whose
    keyword hints don't appear anywhere in the question."""
    keywords = _PROPERTY_KEYWORDS.get(requested_property)
    if not keywords:
        return False
    q = user_question.lower()
    for _fname, name in _detect_products(user_question):
        q = _PRODUCT_STRIP_PATTERNS[name].sub(" ", q)
    return any(_keyword_matches(kw, q) for kw in keywords)


# ---------------------------------------------------------
# 6b. Minimal LLM fallback -- used ONLY when the deterministic rules
#     in analyze_query() below genuinely can't disambiguate a question
#     (see the two call sites there). Asks for exactly two fields, no
#     filters, no field documentation, no examples: the deterministic
#     filters/products already extracted stay as-is either way, this
#     just breaks the intent/requested_property tie.
# ---------------------------------------------------------
_FALLBACK_SYSTEM_PROMPT = (
    'Classify the product-datasheet question. Reply with ONLY minified JSON: '
    '{"intent": "default_qa"|"metadata_list"|"metadata_question"|"comparison", '
    '"requested_property": "<one prop_ field name or empty string>"}'
)


def _analyze_query_llm_fallback(user_question):
    """Returns (intent, requested_property). Never raises -- falls back
    to ("default_qa", "") on any call/parse failure, same as before."""
    try:
        raw = ask_llm(system_prompt=_FALLBACK_SYSTEM_PROMPT, user_prompt=user_question)
        cleaned = raw.strip().strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        intent = parsed.get("intent")
        if intent not in ("default_qa", "metadata_list", "metadata_question", "comparison"):
            raise ValueError(f"unexpected intent {intent!r}")
        requested_property = parsed.get("requested_property") or ""
        if requested_property and not _requested_property_is_plausible(requested_property, user_question):
            requested_property = ""
        return intent, requested_property
    except Exception as e:
        print(f"   [analyze_query] LLM fallback failed, defaulting to default_qa ({e})")
        return "default_qa", ""


# ---------------------------------------------------------
# 6c. analyze_query -- deterministic-first query analysis.
#
#
#     Order of decisions (each one deterministic unless noted):
#       1. Product detection (_detect_products) -- reused everywhere below.
#       2. Comparison: 2+ products mentioned + a comparison cue. If 2+
#          products are mentioned WITHOUT a comparison cue, that's the
#          one genuinely ambiguous case ("mentioned together" vs "asked
#          to compare") -- LLM fallback breaks the tie.
#       3. Non-product filters (su_bazlı/voc_uyumlu/kullanım_alanı/doku)
#          via the existing _deterministic_filters.
#       4. requested_properties: property keywords are treated as a
#          LOOKUP (metadata_question) rather than a FILTER (metadata_list)
#          when either a single product is named (e.g. "AquaLux su bazlı
#          mı?") or the question carries a lookup cue ("nedir"/"mı"/...)
#          without a list cue ("hangi"/"göster"/...).
#       5. If a property keyword is present but step 4's rule can't
#          confidently place it (no product AND cues are absent or
#          contradictory), that's the other ambiguous case -- LLM
#          fallback decides (still just one field; the ambiguous case is
#          rare enough that the ~free deterministic multi-match above is
#          left as the common path and the LLM fallback keeps its
#          existing single-field contract).
#       6. Otherwise: non-product filters with no requested_properties ->
#          metadata_list; nothing structured detected -> default_qa.
#          Both are fully deterministic, no LLM call. metadata_list's
#          displayed label is built from EVERY active filter (boolean
#          filters from _deterministic_filters plus any detected property
#          keywords)
# ---------------------------------------------------------
def _filters_display_label(filters, property_keys):
    """Combined display label for metadata_list, built from every active
    filter criterion (both the boolean/value filters in `filters` and any
    detected property keywords not already covered), joined with ' + '.
    'prop_ürün_adı' is excluded -- naming a product isn't a "criterion"
    worth echoing back in the header. Falls back to "" if nothing has a
    known display label (build_metadata_list_answer defaults that to
    "Eşleşen")."""
    keys = [k for k in filters if k in _PROPERTY_DISPLAY_LABELS]
    for pk in property_keys:
        if pk not in keys:
            keys.append(pk)
    labels = [_PROPERTY_DISPLAY_LABELS.get(k, k) for k in keys]
    return " + ".join(labels)


def analyze_query(user_question):
    """Returns (intent, products, search_query, where, display_property, requested_properties).
    `requested_properties` is always a list (possibly empty) of prop_
    field names -- one entry for a single-field lookup, several for a
    multi-field lookup, empty when the intent isn't a lookup at all."""
    search_query = user_question  # rewriting removed -- always the original text
    products_found = _detect_products(user_question)
    non_product_filters = _deterministic_filters(user_question)

    # --- 1. Comparison -----------------------------------------------
    if len(products_found) >= 2:
        if _COMPARISON_CUE_RE.search(user_question):
            products = [fname for fname, _name in products_found]
            return "comparison", products, search_query, None, "", []
        # 2+ products mentioned but no explicit compare/vs/"arasında"
        # wording -- ambiguous (could be a comparison, could just be two
        # products named in passing). Let the minimal LLM fallback
        # decide; deterministic filters/products stay available either way.
        intent, fallback_property = _analyze_query_llm_fallback(user_question)
        if intent == "comparison":
            products = [fname for fname, _name in products_found]
            return "comparison", products, search_query, None, "", []
        # fall through to build filters/where for any other intent below
    else:
        intent, fallback_property = None, None  # not yet decided

    single_product = products_found[0] if len(products_found) == 1 else None

    if single_product is not None and _COMPARISON_CUE_RE.search(user_question):
        # A single named product plus a comparison cue is an open comparative question,
        # not a structured-field lookup -- single_product alone would otherwise force
        # treat_as_lookup below. Falls to default_qa, still narrowed to
        # this product's own chunks.
        filters = {"prop_ürün_adı": single_product[1]}
        return "default_qa", [], search_query, _filters_to_where(filters), "", []
    
    # --- 2. requested_properties (LOOKUP) vs FILTER ---------------------
    property_keys = _detect_property_keywords(user_question)
    has_lookup_cue = bool(_LOOKUP_CUE_RE.search(user_question))
    has_list_cue = bool(_LIST_CUE_RE.search(user_question))

    treat_as_lookup = bool(property_keys) and (
        single_product is not None or (has_lookup_cue and not has_list_cue)
    )

    if property_keys and not treat_as_lookup and single_product is None \
            and not has_list_cue and not non_product_filters:
        # Property keyword(s) present but nothing (product, clear lookup
        # cue, clear list cue, or another filter) confirms which way to
        # read it -- genuinely ambiguous, ask the small fallback. The
        # fallback only ever names one field; that's fine here since this
        # branch is specifically the case where we couldn't confidently
        # commit to ANY reading deterministically.
        if intent is None:
            intent, fallback_property = _analyze_query_llm_fallback(user_question)
        if intent == "metadata_question" and fallback_property:
            filters = dict(non_product_filters)
            if single_product:
                filters["prop_ürün_adı"] = single_product[1]
            display_property = _PROPERTY_DISPLAY_LABELS.get(fallback_property, fallback_property)
            return "metadata_question", [], search_query, _filters_to_where(filters), display_property, [fallback_property]
        if intent == "metadata_list" or non_product_filters:
            filters = dict(non_product_filters)
            if single_product:
                filters["prop_ürün_adı"] = single_product[1]
            label = _filters_display_label(filters, property_keys)
            return "metadata_list", [], search_query, _filters_to_where(filters), label, []
        return "default_qa", [], search_query, None, "", []

    if treat_as_lookup:
        filters = dict(non_product_filters)
        for pk in property_keys:
            filters.pop(pk, None)  # a field being looked up is never also a filter
        if single_product:
            filters["prop_ürün_adı"] = single_product[1]
        display_property = " + ".join(
            _PROPERTY_DISPLAY_LABELS.get(pk, pk) for pk in property_keys
        )
        return "metadata_question", [], search_query, _filters_to_where(filters), display_property, property_keys
    contains_filters = _contains_filters(user_question)
    # --- 3. metadata_list: non-product filters present, no lookup ------
    if non_product_filters or contains_filters:
        filters = dict(non_product_filters)
        if single_product:
            filters["prop_ürün_adı"] = single_product[1]
        label = _filters_display_label(filters, property_keys)
        return "metadata_list", [], search_query, _filters_to_where(filters), label, []

    # --- 4. Nothing structured detected: plain document question -------
    # (A bare product-name mention with no other filter/property stays
    # default_qa -- e.g. "AquaLux ... hangi astar önerilir?" is a
    # genuine document question, not a list/lookup -- but the product
    # name still narrows retrieval to that product's own chunks.)
    filters = {"prop_ürün_adı": single_product[1]} if single_product else {}
    return "default_qa", [], search_query, _filters_to_where(filters), "", []


def _filters_to_where(filters):
    """{'prop_doku': 'mat', 'prop_sarfiyat_max': {'$lte': 10}} -> Chroma
    `where` clause. Bare values become equality; dict values (e.g.
    {'$lte': 10}) pass through as-is. None if there's nothing to filter."""
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if value is None:
            continue
        if value == "":
            continue
        clauses.append({key: value} if isinstance(value, dict) else {key: {"$eq": value}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _record_matches_where(metadata, where):
    """Minimal local evaluator for the same `where` shape, used to filter
    the in-memory BM25 candidate pool (Chroma's `where` only applies
    server-side to `collection.query`/`.get`, not to our own bm25 index)."""
    if where is None:
        return True
    if "$and" in where:
        return all(_record_matches_where(metadata, clause) for clause in where["$and"])
    (key, cond), = where.items()
    if key not in metadata:
        return False
    actual = metadata[key]
    if isinstance(cond, dict):
        op, expected = next(iter(cond.items()))
        if op == "$eq":
            return actual == expected
        if op == "$ne":
            return actual != expected
        if op == "$gt":
            return actual > expected
        if op == "$gte":
            return actual >= expected
        if op == "$lt":
            return actual < expected
        if op == "$lte":
            return actual <= expected
        return False
    return actual == cond


# ---------------------------------------------------------
# 7a. Retrieval mode: semantic-only (dense vector) search.
#     Returns a ranked list of {"id", "text", "metadata"} records.
# ---------------------------------------------------------
def semantic_search(user_question, collection, final_n=FINAL_TOP_K, where=None):
    query_vector = get_embedding(user_question)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=final_n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [{"id": i, "text": d, "metadata": m, "distance": dist} for i, d, m, dist in zip(ids, documents, metadatas, distances)]


# ---------------------------------------------------------
# 7b. Retrieval mode: hybrid (vector + BM25, fused with RRF).
#     Returns a ranked list of {"id", "text", "metadata"} records.
# ---------------------------------------------------------
def hybrid_search(user_question, collection, bm25_index, final_n=FINAL_TOP_K, where=None,
                   vector_top_k=VECTOR_TOP_K, bm25_top_k=BM25_TOP_K):
    # --- Vector (dense) search --- (Chroma applies `where` server-side)
    query_vector = get_embedding(user_question)
    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=vector_top_k,
        where=where,
        include=["metadatas", "distances"],
    )
    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_ranked_ids = vector_ids  # no filtering, just ranking as before

    # --- BM25 (sparse/keyword) search --- (apply `where` locally first)
    tokenized_query = tokenize(user_question)
    bm25_scores = bm25_index["bm25"].get_scores(tokenized_query)
    eligible_indices = [
        i for i in range(len(bm25_scores))
        if _record_matches_where(bm25_index["metadatas"][i], where)
    ]
    ranked_indices = sorted(eligible_indices, key=lambda i: bm25_scores[i], reverse=True)[:bm25_top_k]
    bm25_ranked_ids = [bm25_index["ids"][i] for i in ranked_indices]

    # --- Fuse the two rankings ---
    fused_ids = reciprocal_rank_fusion([vector_ranked_ids, bm25_ranked_ids])
    top_ids = fused_ids[:final_n]

    # Build lookup dicts so we can attach scores to the final records
    dense_score_map = dict(zip(vector_ids, vector_distances))
    sparse_score_map = {bm25_index["ids"][i]: bm25_scores[i] for i in ranked_indices}

    fetched = collection.get(ids=top_ids, include=["documents", "metadatas"])
    id_to_record = {
        doc_id: {
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "dense_distance": dense_score_map.get(doc_id),
            "sparse_score": sparse_score_map.get(doc_id),
        }
        for doc_id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    }

    return [id_to_record[doc_id] for doc_id in top_ids if doc_id in id_to_record]


# ---------------------------------------------------------
# 7c. Strategy for "default_qa": keep hybrid retrieval, but widen the
#     candidate pool and then collapse to ONE (highest-ranked) chunk per
#     parent, so one document's many chunks can't crowd out others.
# ---------------------------------------------------------
def dedupe_by_parent(records, keep_n=None):
    """`records` is already ranked (RRF/vector order) -- keep the first
    (best) record seen for each parent_id, or `source` as a fallback key
    for chunks that don't carry a parent_id (e.g. parent-child disabled)."""
    seen = set()
    deduped = []
    for r in records:
        key = r["metadata"].get("parent_id") or r["metadata"].get("source")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped[:keep_n] if keep_n else deduped


def _contains_filters(user_question):
    """Fields needing a substring/contains match against a metadata value
    stored as a joined list (e.g. prop_ambalaj_boyutları = "1, 5, 15")
    rather than exact equality. Chroma's `where` can't do partial-string
    matching on metadata, so these are applied as a local post-filter in
    metadata_list_search, not folded into the Chroma where clause."""
    q = user_question.lower()
    filters = {}
    if re.search(r"ambalaj|paket", q):
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*litre", q)
        if m:
            filters["prop_ambalaj_boyutları"] = m.group(1).replace(",", ".")
    return filters


def _meta_matches_contains(metadata, contains_filters):
    for key, needle in contains_filters.items():
        value = str(metadata.get(key, ""))
        tokens = [t.strip() for t in value.split(",")]
        if needle not in tokens:
            return False
    return True
# ---------------------------------------------------------
# 7d. Strategy for "metadata_question": NEVER runs vector similarity
#     search. Filters are turned into a Chroma `where` clause and passed
#     to collection.get(...), which returns every matching child chunk
#     with no ranking/truncation involved. Then we deduplicate down to
#     one representative record per SOURCE DOCUMENT (expanded to that
#     chunk's parent for a fuller excerpt) so every matching product gets
#     exactly one slot in the final context, regardless of how many
#     chunks of it matched. This one representative excerpt is what still
#     gets handed to the LLM afterwards (in retrieve()/answer_with_context) --
#     unlike metadata_list below, which never touches the LLM at all.
# ---------------------------------------------------------
def metadata_question_search(where, collection, parent_collection):
    if where is None:
        return []

    matched = collection.get(where=where, include=["documents", "metadatas"])

    by_source = {}
    for doc_id, doc, meta in zip(matched["ids"], matched["documents"], matched["metadatas"]):
        source = meta["source"]
        if source not in by_source:
            by_source[source] = {"id": doc_id, "text": doc, "metadata": meta}

    parent_ids = [rec["metadata"]["parent_id"] for rec in by_source.values() if "parent_id" in rec["metadata"]]
    if not parent_ids:
        return list(by_source.values())

    parents = parent_collection.get(ids=parent_ids, include=["documents", "metadatas"])
    parent_map = {pid: (doc, meta) for pid, doc, meta in zip(parents["ids"], parents["documents"], parents["metadatas"])}

    records = []
    for rec in by_source.values():
        pid = rec["metadata"].get("parent_id")
        if pid in parent_map:
            doc, meta = parent_map[pid]
            records.append({"id": pid, "text": doc, "metadata": meta})
        else:
            records.append(rec)
    return records


# ---------------------------------------------------------
# 7e. Strategy for "metadata_list": a pure structured-database lookup.
#     NEVER calls collection.query(), NEVER generates embeddings, NEVER
#     runs BM25, and NEVER expands to parent chunks -- listing questions
#     ("Hangi boyalar su bazlıdır?") aren't nearest-neighbor questions or
#     document Q&A, they're "which rows match this filter", so this stays
#     entirely inside collection.get(where=...).
#
#     Every matching CHILD chunk is fetched, then deduplicated by product
#     name (metadata["prop_ürün_adı"] -- NOT source/parent_id, since one
#     product can have many chunks) so each product appears exactly once.
# ---------------------------------------------------------
def metadata_list_search(where, collection, user_question=None):
    matched = collection.get(where=where, include=["metadatas"])
    contains_filters = _contains_filters(user_question) if user_question else {}

    by_product = {}
    for meta in matched["metadatas"]:
        product_name = meta.get("prop_ürün_adı")
        if not product_name or product_name in by_product:
            continue
        if contains_filters and not _meta_matches_contains(meta, contains_filters):
            continue
        by_product[product_name] = {
            "product_name": product_name,
            "source": meta.get("source"),
            "page": meta.get("page_number"),
        }

    return list(by_product.values())


# ---------------------------------------------------------
# 7e. Generic answer formatter for "metadata_list". Never sent through
#     the LLM and never hardcodes a property name -- the metadata database
#     already contains the answer, so we just render it directly.
#
#     Works for ANY property: build_metadata_list_answer(products, "Su bazlı")
#     and build_metadata_list_answer(products, "VOC uyumlu") both produce
#     the same "<display_property> ürünler:\n\n• Product\n• Product" shape.
# ---------------------------------------------------------
def build_metadata_list_answer(products, display_property):
    label = display_property or "Eşleşen"

    if not products:
        return f"{label} ürün bulunamadı."

    lines = [f"{label} ürünler:", ""]
    for product in sorted(products, key=lambda p: p["product_name"]):
        lines.append(f"• {product['product_name']}")
    return "\n".join(lines)

def _lookup_single_property_value(metadata, requested_property):
    """Reads one structured field's value out of an already-fetched
    metadata dict. For range-shaped fields -- stored by pdf_extraction.py
    as sibling "<root>_min" / "<root>_max" keys sharing ONE
    "<root>_birimi" unit key (never "<root>_min_birimi") -- this strips
    the _min/_max suffix to find the right unit key, and merges both
    bounds into a "min-max unit" answer instead of silently returning
    just one arbitrary bound (e.g. "prop_sarfiyat_min" alone). Returns
    None if the field has no value for this product."""
    root, bound = requested_property, None
    if root.endswith("_min"):
        root, bound = root[:-4], "min"
    elif root.endswith("_max"):
        root, bound = root[:-4], "max"

    unit = metadata.get(f"{root}_birimi")
    if unit is None:
        # not actually a range field -- fall back to the naive pattern
        unit = metadata.get(f"{requested_property}_birimi")

    if bound is not None:
        min_val = metadata.get(f"{root}_min")
        max_val = metadata.get(f"{root}_max")
        if min_val is not None and max_val is not None:
            value = min_val if min_val == max_val else f"{min_val}-{max_val}"
        else:
            value = metadata.get(requested_property)
    else:
        value = metadata.get(requested_property)

    if value is None:
        return None
    if unit is not None:
        return f"{value} {unit}"
    return value


def metadata_question_lookup(where, requested_properties, collection):
    """Looks up one or more structured fields' values for the matched
    product and returns them as a single combined, labeled answer string
    (one "<Label>: <value>" line per field, in the order requested) --
    or None if the product can't be found or NONE of the requested
    fields have a value (so the caller can fall back to
    metadata_question_search / the LLM instead of returning an empty
    answer). Fields that individually have no value are silently
    skipped rather than failing the whole lookup, so "sarfiyatı ve
    depolama süresi nedir?" still answers with whichever of the two the
    document actually states."""
    if where is None or not requested_properties:
        return None
    matched = collection.get(where=where, limit=1, include=["metadatas"])
    if not matched["ids"]:
        return None
    metadata = matched["metadatas"][0]

    lines = []
    for requested_property in requested_properties:
        value = _lookup_single_property_value(metadata, requested_property)
        if value is None:
            continue
        label = _PROPERTY_DISPLAY_LABELS.get(requested_property, requested_property)
        lines.append(f"{label}: {value}")

    if not lines:
        return None
    return "\n".join(lines)

# ---------------------------------------------------------
# 7f. Strategy for "comparison": one representative PARENT chunk per
#     referenced product, fetched directly by `source` -- no nearest-
#     neighbor ranking involved, so every named product is guaranteed a
#     slot instead of competing for it.
# ---------------------------------------------------------
def comparison_search(products, collection, parent_collection):
    records = []
    for source in products:
        matched = collection.get(where={"source": {"$eq": source}}, limit=1, include=["documents", "metadatas"])
        if not matched["ids"]:
            continue
        child_meta = matched["metadatas"][0]
        parent_id = child_meta.get("parent_id")
        if parent_id:
            parents = parent_collection.get(ids=[parent_id], include=["documents", "metadatas"])
            if parents["ids"]:
                records.append({"id": parent_id, "text": parents["documents"][0], "metadata": parents["metadatas"][0]})
                continue
        records.append({"id": matched["ids"][0], "text": matched["documents"][0], "metadata": child_meta})
    return records


# ---------------------------------------------------------
# 8a. Context builder A: "flat" — use the matched CHILD chunk text
#     directly as context (no parent expansion).
# ---------------------------------------------------------
def build_context(records):
    pages = [r["metadata"]["page_number"] for r in records]
    context = "\n\n".join(
        f"Source: {r['metadata']['source']}\n(Page {r['metadata']['page_number']}): {_confidence_tag(r)}{r['text']}"
        for r in records
    )
    for i, r in enumerate(records):
        print(f"\n===== Chunk {i+1} =====")
        print(f"Source: {r['metadata']['source']}")
        print(f"Page: {r['metadata']['page_number']}")
        if "distance" in r:
            print(f"Dense distance: {r['distance']:.4f}")
        if r.get("dense_distance") is not None:
            print(f"Dense distance: {r['dense_distance']:.4f}")
        if r.get("sparse_score") is not None:
            print(f"Sparse score: {r['sparse_score']:.4f}")
        if r.get("confidence") is not None:
            print(f"Confidence: {r['confidence']:.4f} ({r['confidence_label']})")
        print(r["text"])
    return context, pages


# ---------------------------------------------------------
# 8b. Context builder B: "parent_child" — expand matched child chunks
#     up to their PARENT chunk text as context.
# ---------------------------------------------------------
def build_context_parent_child(records, parent_collection):
    matched_parent_ids = [r["metadata"]["parent_id"] for r in records]
    unique_parent_ids = list(dict.fromkeys(matched_parent_ids))

    parents = parent_collection.get(ids=unique_parent_ids, include=["documents", "metadatas"])
    parent_pages = [m["page_number"] for m in parents["metadatas"]]


    parent_map = {
        pid: (doc, meta)
        for pid, doc, meta in zip(
            parents["ids"],
            parents["documents"],
            parents["metadatas"]
        )
    }

    contexts = []

    for record in records:
        pid = record["metadata"]["parent_id"]
        doc, meta = parent_map[pid]
        tag = _confidence_tag(record)

        contexts.append(
            f"""Source: {meta['source']}
    Page: {meta['page_number']}

    {tag}{doc}
    """
        )

    context = "\n\n".join(contexts)
    return context, parent_pages


# ---------------------------------------------------------
# 8c. Context builder C: "generic" -- for records that are already the
#     final thing to show (parent chunks from metadata_question/comparison,
#     or deduped child chunks from default_qa). No parent expansion,
#     no dense/sparse score assumptions.
# ---------------------------------------------------------
def build_context_generic(records):
    pages = [r["metadata"].get("page_number") for r in records]
    context = "\n\n".join(
        f"Source: {r['metadata']['source']}\n(Page {r['metadata'].get('page_number')}): {_confidence_tag(r)}{r['text']}"
        for r in records
    )
    return context, pages


# ---------------------------------------------------------
# 9a. Intent-routed retrieval entry point. Classifies the question
#     first, THEN picks the retrieval strategy that actually fits it --
#     this is what replaces "always run vector search" as the default.
#
#     Returns (intent, context, pages, records, direct_answer).
#     `direct_answer` is None for every intent except "metadata_list",
#     where it's the final answer string already generated in Python --
#     the caller must use it as-is and must NOT send it to the LLM.
# ---------------------------------------------------------
def retrieve(user_question, collection, parent_collection, bm25_index):
    intent, products, search_query, where, display_property, requested_properties = analyze_query(user_question)
    if intent == "comparison" and products:
        records = comparison_search(products, collection, parent_collection)
        context, pages = build_context_generic(records)
        return intent, context, pages, records, None

    if intent == "metadata_list":
        products_matched = metadata_list_search(where, collection, user_question)
        direct_answer = build_metadata_list_answer(products_matched, display_property)
        # context/pages are built too (for the debug print + consistent
        # return shape) but they're never handed to an LLM for this intent.
        context, pages = build_context_generic([
            {
                "id": p["product_name"],
                "text": p["product_name"],
                "metadata": {"source": p["source"], "page_number": p["page"]},
            }
            for p in products_matched
        ])
        return intent, context, pages, products_matched, direct_answer

    if intent == "metadata_question" and requested_properties:
        # metadata_question_lookup already returns a fully labeled answer
        # (one "Label: value" line per requested field, multiple fields
        # joined with newlines), so it's used as-is -- no re-wrapping.
        direct_answer = metadata_question_lookup(where, requested_properties, collection)
        if direct_answer is not None:
            return intent, "", [], [], direct_answer

    if intent == "metadata_question" and where is not None:
        records = metadata_question_search(where, collection, parent_collection)
        context, pages = build_context_generic(records)
        return intent, context, pages, records, None

    # Fallback to default_qa
    intent = "default_qa"
    records = hybrid_search(
        search_query, collection, bm25_index,
        final_n=DEFAULT_QA_CANDIDATE_K, where=where,
        vector_top_k=DEFAULT_QA_CANDIDATE_K, bm25_top_k=DEFAULT_QA_CANDIDATE_K,
    )
    # Gate on confidence BEFORE dedupe/truncation, over the wide
    # candidate pool -- so a dropped low-confidence chunk simply lets
    # the next-best parent take its slot in FINAL_TOP_K, instead of
    # thresholding an already-final top-5 and silently shrinking it.
    records = apply_confidence_thresholds(records)
    records = dedupe_by_parent(records, keep_n=FINAL_TOP_K)
    context, pages = build_context_parent_child(records, parent_collection)
    return intent, context, pages, records, None

# ---------------------------------------------------------
# 9b. LLM call for default RAG process. If direct_answer == None
#     then it directs here.
# ---------------------------------------------------------
def answer_with_context(user_question, context):
    system_prompt = f"""You are a document assistant.

Use ONLY the provided source excerpts.

The user's input may be:
- a question,
- a product name,
- a keyword,
- a specification,
- or a short phrase.

If the input is not a complete question, treat it as a request to summarize or explain the matching information from the excerpts or return names of matching documents.

Some excerpts are prefixed with "[DÜŞÜK GÜVEN - ...]" -- this means the
retrieval system is not confident this excerpt actually matches the
question. You may still use it, but hedge explicitly (e.g. "Bu konuda
emin değilim, ancak..." / "Belgede bulunan bir ifadeye göre..." rather
than stating it as settled fact). Never present a "DÜŞÜK GÜVEN" excerpt
with the same certainty as an untagged one.

If the excerpts contain the requested information:
- answer clearly,
- combine relevant excerpts when appropriate,
- cite only the Source and Page information explicitly present in the provided excerpts.
- Never invent URLs, hyperlinks, document locations, page numbers, section names, or citations.

If the excerpts do not contain the requested information, reply exactly:

"This information is not found in the provided excerpts."

Never use knowledge outside the excerpts.
Never fabricate missing information.
"""
    user_prompt = f"""
User request:
{user_question}

Relevant source excerpts:
{context}

Respond according to the system instructions.
"""

    return ask_llm(system_prompt=system_prompt, user_prompt=user_prompt)


# ---------------------------------------------------------
# 10. Store in Chroma DB and run the RAG query loop
# ---------------------------------------------------------
def main():
    print("Step 1: Setting up Chroma DB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    parent_collection = chroma_client.get_or_create_collection(name=f"{COLLECTION_NAME}_parents")

    existing_count = collection.count()
    print(f"   Collection currently has {existing_count} child chunks.\n")

    print("Step 2: Checking which files still need to be ingested...")
    for pdf_file in PDF_FILES:
        pdf_path = os.path.join(DATASET_PATH, pdf_file)
        already_ingested = collection.get(where={"source": pdf_file}, limit=1)["ids"]
        if already_ingested:
            print(f"   '{pdf_path}' already ingested — skipping.")
        else:
            print(f"   '{pdf_path}' not found in collection — ingesting now...")
            ingest_pdf(pdf_path, collection, parent_collection, pdf_file)

    print(f"Collection now has {collection.count()} child chunks and {parent_collection.count()} parent chunks in '{COLLECTION_NAME}'.\n")

    print("Step 3: Building BM25 keyword index over child chunks...")
    bm25_index = build_bm25_index(collection)
    print(f"   BM25 index built over {len(bm25_index['ids'])} child chunks.\n")

    print(f"Active SEARCH_MODE: {SEARCH_MODE}")
    print("Ready! Ask questions about the documents (type 'exit'/'quit' to stop).")

    while True:
        raw_input_text = input("Your question: ").strip()
        if raw_input_text.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break


        user_question = raw_input_text
        start = time.perf_counter()
        intent, context, pages, records, direct_answer = retrieve(user_question, collection, parent_collection, bm25_index)
        elapsed = time.perf_counter() - start

        print(f"   [intent] {intent}")


        # metadata_list already has its final answer generated in Python --
        # the database already contains the answer, so it never touches the LLM.
        answer = direct_answer if direct_answer is not None else answer_with_context(user_question, context)

        print("\n--- Answer ---")
        print(answer)
        print(f"(Intent: {intent} | retrieval time: {elapsed:.3f}s)\n")
        print("Sources:\n")
        for r in records:
            if "metadata" in r:
                print(r["metadata"].get("parent_id", r["id"]))
            else:
                print(r.get("product_name"))



if __name__ == "__main__":
    main()