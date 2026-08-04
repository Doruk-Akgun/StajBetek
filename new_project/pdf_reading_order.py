import fitz  # PyMuPDF
from collections import Counter

BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit 4 (=16) is bold.
# NOTE: bit 1 (=2) is italic, not bold -- verified against real spans in a
# test document (Arial,Bold spans reported flags=16; italic never appears).


# ---------------------------------------------------------------------------
# 1. Font/style metadata (from page.get_text("dict")), tied to word geometry
# ---------------------------------------------------------------------------

def _collect_font_sizes(page):
    """(size, char_count) for every non-empty span on the page."""
    sizes = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # skip images/non-text blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    sizes.append((round(span["size"], 1), len(text.strip())))
    return sizes


def _body_font_size(all_sizes):
    weighted = Counter()
    for size, char_count in all_sizes:
        weighted[size] += char_count
    if not weighted:
        return 10.0
    return weighted.most_common(1)[0][0]


def _all_spans_style(page):
    """(bbox, size, is_bold) for every non-empty span on the page."""
    styles = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                is_bold = bool(span.get("flags", 0) & BOLD_FLAG)
                styles.append((tuple(span["bbox"]), span["size"], is_bold))
    return styles


def _bbox_overlap_ratio(word_bbox, span_bbox):
    """Fraction of `word_bbox`'s area covered by `span_bbox`."""
    ax0, ay0, ax1, ay1 = word_bbox
    bx0, by0, bx1, by1 = span_bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(ax1 - ax0, 1e-6) * max(ay1 - ay0, 1e-6)
    return inter / area


def _dedupe_words(raw_words, iou_threshold=0.6):
    """Drops duplicate word tuples (same text, near-identical bbox).
    Some PDFs (e.g. templates with a shadow/duplicate text layer) emit
    the same word twice at nearly the same position; left as-is, this
    doubles words in both body text AND headings (e.g. 'UYGULAMA
    UYGULAMA ÖNERİLERİ ÖNERİLERİ'). Dedup happens here, on the raw word
    list, so it's invisible to the XY-cut splitting logic downstream --
    it just never sees the duplicate in the first place.

    Uses IoU (intersection-over-union) rather than a per-corner distance
    tolerance: duplicate-layer glyphs are usually near-identical on
    three edges but can drift several points on one edge (e.g. x1) if
    the duplicate renders at a slightly different weight/kerning --
    checked against a real case where x0/y0/y1 matched within ~0.1pt
    but x1 differed by 4.7pt, large enough to dodge a flat per-corner
    tolerance. IoU tolerates that: an occasional stretched edge doesn't
    drag total overlap below threshold when the rest of the box matches
    this closely. Grouping candidates by exact text first keeps this
    close to O(n) in practice (few words share exact text on one page)."""
    by_text = {}
    for idx, w in enumerate(raw_words):
        by_text.setdefault(w[4], []).append(idx)

    keep = [True] * len(raw_words)
    for idx_list in by_text.values():
        kept_boxes = []
        for idx in idx_list:
            box = raw_words[idx][:4]
            is_dup = any(_bbox_overlap_ratio(box, kb) >= iou_threshold
                         and _bbox_overlap_ratio(kb, box) >= iou_threshold
                         for kb in kept_boxes)
            if is_dup:
                keep[idx] = False
            else:
                kept_boxes.append(box)

    return [w for i, w in enumerate(raw_words) if keep[i]]


def _get_words_with_style(page, spans_style, default_size, overlap_threshold=0.5):
    """page.get_text("words") word tuples, each tagged with the size/bold
    of whichever "dict" span best overlaps its bounding box -- matched by
    geometry, not text, since the two extraction passes tokenize
    independently and text-matching would be fragile (whitespace/encoding
    differences). Falls back to (default_size, not bold) if no span
    overlaps enough; that should be rare, but a safe default beats a
    crash or a mismatched style."""
    raw_words = page.get_text("words")  # (x0,y0,x1,y1,text,block_no,line_no,word_no)
    raw_words = _dedupe_words(raw_words)
    items = []
    for w in raw_words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        best_ratio, best_size, best_bold = 0.0, default_size, False
        for span_bbox, size, is_bold in spans_style:
            ratio = _bbox_overlap_ratio((x0, y0, x1, y1), span_bbox)
            if ratio > best_ratio:
                best_ratio, best_size, best_bold = ratio, size, is_bold
        if best_ratio < overlap_threshold:
            best_size, best_bold = default_size, False
        items.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text,
                       "size": best_size, "is_bold": best_bold})
    return items


# ---------------------------------------------------------------------------
# 2. Recursive XY-cut (geometry/splitting logic unchanged)
# ---------------------------------------------------------------------------

def _project_gaps(intervals, lo, hi, min_gap):
    intervals = sorted(intervals)
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps = []
    prev_end = lo
    for s, e in merged:
        if s - prev_end > min_gap:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if hi - prev_end > min_gap:
        gaps.append((prev_end, hi))
    return gaps


def _leading_bold_prefix(line_words, max_prefix_words=8, max_prefix_chars=60):
    """If a line starts with a contiguous run of bold words ending in
    ':' and is followed by at least one non-bold word, returns
    (prefix_text, rest_text); otherwise (None, None).

    Catches inline sub-headings like 'Ham Sıvalı Yüzey: Öncelikle sıva
    yüzeyinin...' -- a bold label glued onto the same physical line as
    its own body text. Line-grouping correctly keeps that as ONE line
    (same baseline), but that means the merged text is paragraph-length
    and the whole-line heading scorer would never flag it: it trips
    HEADING_MAX_LEN before scoring even runs, and its bold_ratio is a
    small fraction rather than a clean signal either way. This is
    checked at the word level, before lines get merged into a single
    text blob, which is the only place the bold/non-bold boundary is
    still visible.

    Guards against misfiring on a line that's bold in its entirety
    (run_end == len(line_words) is excluded -- that's a normal whole-
    line heading, handled by the regular scorer) and against long
    "prefixes" that are really just an ALL-bold sentence that happens
    to contain a colon somewhere past a reasonable label length.
    """
    if not line_words or not line_words[0]["is_bold"]:
        return None, None
    run_end = 0
    for w in line_words:
        if w["is_bold"]:
            run_end += 1
        else:
            break
    if run_end == 0 or run_end >= len(line_words):
        return None, None
    prefix_words = line_words[:run_end]
    prefix_text = " ".join(w["text"] for w in prefix_words).strip()
    if not prefix_text.endswith(":"):
        return None, None
    if len(prefix_words) > max_prefix_words or len(prefix_text) > max_prefix_chars:
        return None, None
    rest_text = " ".join(w["text"] for w in line_words[run_end:]).strip()
    if not rest_text:
        return None, None
    return prefix_text, rest_text


def _group_into_lines(items, tol=2.5):
    """Group word-items into lines using y-tolerance clustering (a word
    joins an existing line if its y0 is within `tol` points of that
    line's reference y0 -- avoids fixed-bucket edge bugs). Each resulting
    line carries its bbox and enough style info (bold_ratio, dominant
    font size) for heading scoring later; block_margin is filled in by
    the caller, since it depends on ALL lines from the same XY-cut leaf."""
    lines = []  # list of [ref_y, [word-items]]
    for w in sorted(items, key=lambda it: (it["y0"], it["x0"])):
        placed = False
        for entry in lines:
            if abs(w["y0"] - entry[0]) <= tol:
                entry[1].append(w)
                placed = True
                break
        if not placed:
            lines.append([w["y0"], [w]])

    lines.sort(key=lambda entry: entry[0])
    result = []
    for _, line_words in lines:
        line_words.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line_words)
        bbox = (
            min(w["x0"] for w in line_words),
            min(w["y0"] for w in line_words),
            max(w["x1"] for w in line_words),
            max(w["y1"] for w in line_words),
        )
        bold_ratio = sum(1 for w in line_words if w["is_bold"]) / len(line_words)
        size_counter = Counter(round(w["size"], 1) for w in line_words)
        dominant_size = size_counter.most_common(1)[0][0]
        entry = {"text": text, "bbox": bbox, "bold_ratio": bold_ratio,
                 "dominant_size": dominant_size}
        prefix_text, rest_text = _leading_bold_prefix(line_words)
        if prefix_text is not None:
            entry["inline_heading_prefix"] = prefix_text
            entry["inline_heading_rest"] = rest_text
        result.append(entry)
    return result


def _xy_cut_to_lines(items, min_gap_x=5, min_gap_y=3, depth=0, max_depth=10, line_tol=2.5, x_gap_size_ratio=0.8):
    """Recursively splits `items` (word-dicts) along the widest empty
    vertical gap (columns), then the widest empty horizontal gap (stacked
    blocks). Returns an ORDERED LIST of line-dicts plus break markers
    ({"break": "col"} / {"break": "row"}), so heading scoring, bullet
    handling, and cross-page boilerplate stripping can run on top of it."""
    if not items:
        return []

    xs = [(it["x0"], it["x1"]) for it in items]
    ys = [(it["y0"], it["y1"]) for it in items]
    lo_x, hi_x = min(it["x0"] for it in items), max(it["x1"] for it in items)
    lo_y, hi_y = min(it["y0"] for it in items), max(it["y1"] for it in items)

    if depth < max_depth:
        max_height = max((it["y1"] - it["y0"] for it in items), default=0)
        num_lines = len(_group_into_lines(items, tol=line_tol))
        y_span = hi_y - lo_y
        if num_lines <= 2 or y_span > 200:
            effective_min_gap_x = max(min_gap_x, x_gap_size_ratio * max_height)
        else:
            effective_min_gap_x = max(min_gap_x / 2, 2.0)
        x_gaps = _project_gaps(xs, lo_x, hi_x, effective_min_gap_x)
        x_gaps = [g for g in x_gaps if g[0] > lo_x + 1 and g[1] < hi_x - 1]
        if x_gaps:
            gap = max(x_gaps, key=lambda g: g[1] - g[0])
            split_x = (gap[0] + gap[1]) / 2
            left = [it for it in items if (it["x0"] + it["x1"]) / 2 < split_x]
            right = [it for it in items if (it["x0"] + it["x1"]) / 2 >= split_x]
            if left and right:
                left_lines = _xy_cut_to_lines(left, min_gap_x, min_gap_y, depth + 1, max_depth, line_tol, x_gap_size_ratio)
                right_lines = _xy_cut_to_lines(right, min_gap_x, min_gap_y, depth + 1, max_depth, line_tol, x_gap_size_ratio)
                return left_lines + [{"break": "col"}] + right_lines

    if depth < max_depth:
        y_gaps = _project_gaps(ys, lo_y, hi_y, min_gap_y)
        y_gaps = [g for g in y_gaps if g[0] > lo_y + 1 and g[1] < hi_y - 1]
        if y_gaps:
            gap = max(y_gaps, key=lambda g: g[1] - g[0])
            split_y = (gap[0] + gap[1]) / 2
            top = [it for it in items if (it["y0"] + it["y1"]) / 2 < split_y]
            bottom = [it for it in items if (it["y0"] + it["y1"]) / 2 >= split_y]
            if top and bottom:
                top_lines = _xy_cut_to_lines(top, min_gap_x, min_gap_y, depth + 1, max_depth, line_tol, x_gap_size_ratio)
                bottom_lines = _xy_cut_to_lines(bottom, min_gap_x, min_gap_y, depth + 1, max_depth, line_tol, x_gap_size_ratio)
                return top_lines + [{"break": "row"}] + bottom_lines

    # Base case: this group of words is one coherent column/paragraph
    # block. Group into lines, then tag every line with this block's
    # left margin (the most common line-start x, i.e. where the body
    # text of THIS block normally begins) for the heading scorer.
    lines = _group_into_lines(items, tol=line_tol)
    if lines:
        margin = Counter(round(l["bbox"][0]) for l in lines).most_common(1)[0][0]
        for l in lines:
            l["block_margin"] = margin
    return lines


# ---------------------------------------------------------------------------
# 3. Weighted-scoring heading detection
# ---------------------------------------------------------------------------

HEADING_MAX_LEN = 50  # a line longer than this is prose, never a heading
# (verified against a real multi-section spec sheet: genuine headings/
# labels topped out at 36 chars; a false-positive intro-paragraph line
# that scored as a heading was 83 chars -- 50 leaves clear margin.)


def _score_and_flag_headings(line_items, body_size, size_ratio=1.1,
                              margin_tol=3.0, gap_factor=1.6, threshold=6):
    """Scores each line and flags it as a heading if the score clears
    `threshold`. Runs once, in reading order, over a whole page's
    line/break list, since several features are inherently about a
    line's position relative to its NEIGHBORS (paragraph start,
    mid-sentence, vertical whitespace) -- they can't be judged from a
    single line in isolation.

    Score weights (mirrors the spec this was designed against):
        bold                                +3
        larger than body text                +2
        starts at its block's left margin    +2
        starts a new paragraph                +2   (mutually exclusive:
        appears mid-sentence                 -3    exactly one applies)
        ends with ':'                         +2
        ALL CAPS                              +1
        appears after vertical whitespace     +1
        shares its line with non-bold body
          text (mixed formatting on 1 line)   -2   (waived if it also
                                                     starts at the margin
                                                     AND ends with ':')

    Implementation choices, and why:
    - "starts a new paragraph" / "mid-sentence" are opposites here, not
      independent flags -- a line either continues the previous
      sentence or it doesn't. Treating them as mutually exclusive avoids
      double-counting and matches the spec's clearly opposed weights
      (+2 vs -3).
    - "after vertical whitespace" is measured two ways: an explicit
      XY-cut break (real column/block boundary), or a y-gap to the
      previous line bigger than ~1.6x that line's font size (a
      reasonable proxy for "extra leading before a new block" without
      needing a document-wide line-height model).
    - "starts at margin" compares against the block's own typical
      left-start x (computed in _xy_cut_to_lines), not the block's
      bounding-box edge -- an indented or centered heading shouldn't be
      the reference point for what counts as "the margin".
    """
    prev_line = None
    prev_was_break = True  # start of a page/block counts as a break

    for item in line_items:
        if item.get("break"):
            prev_was_break = True
            continue

        text = item["text"].strip()
        if not text:
            continue

        if not any(c.isalpha() for c in text):
            # Symbol/number-only lines (e.g. a big decorative "+%20" in an
            # infographic box) can still be bold/large/isolated enough to
            # score above threshold, but they're never real headings --
            # a heading always contains at least one letter.
            item["is_heading"] = False
            prev_line = item
            prev_was_break = False
            continue

        has_real_word = any(
            len(tok) >= 3 and any(c.isalpha() for c in tok)
            for tok in text.replace(":", " ").split()
        )
        if not has_real_word:
            # A lone unit/number label (e.g. "2.5 L" under a package
            # icon) can be isolated + margin-aligned + technically
            # "all caps" (a single uppercase letter like "L" satisfies
            # that check) -- the same shape a real heading has. Since
            # the alpha check above only requires ONE letter anywhere
            # in the line, "2.5 L" still gets past it. Requiring one
            # real (3+ letter) word filters these out without touching
            # genuine short headings like "TEK KAT" or "EK BİLGİLER".
            item["is_heading"] = False
            prev_line = item
            prev_was_break = False
            continue

        if item.get("inline_heading_prefix"):
            # Confirmed at the word level during grouping (bold run
            # ending in ':' followed by non-bold text) -- skip the
            # whole-line score entirely, since the merged text here is
            # paragraph-length and would otherwise trip HEADING_MAX_LEN.
            item["is_heading"] = True
            prev_line = item
            prev_was_break = False
            continue

        if len(text) > HEADING_MAX_LEN:
            # Real headings/labels are short. A long line -- even in a
            # larger, styled "intro paragraph" font -- is prose, not a
            # heading; the position/style features alone can't tell
            # those apart (verified against a real doc: a 10pt subtitle
            # sentence otherwise scored as a heading here).
            item["is_heading"] = False
            prev_line = item
            prev_was_break = False
            continue

        bold = item.get("bold_ratio", 0.0) >= 0.6
        larger = item.get("dominant_size", body_size) >= body_size * size_ratio
        margin = item.get("block_margin")
        starts_margin = margin is not None and abs(item["bbox"][0] - margin) <= margin_tol
        ends_colon = text.endswith(":")
        allcaps = text.isupper() and any(c.isalpha() for c in text)
        mixed_line = 0.0 < item.get("bold_ratio", 0.0) < 1.0

        vertical_gap = False
        if not prev_was_break and prev_line is not None:
            gap = item["bbox"][1] - prev_line["bbox"][3]
            line_height = prev_line.get("dominant_size", body_size)
            vertical_gap = gap > line_height * gap_factor
        after_whitespace = prev_was_break or vertical_gap

        prev_ended_sentence = (
            prev_line is not None
            and prev_line["text"].rstrip().endswith(('.', ':', ';', '!', '?'))
        )
        starts_new_paragraph = after_whitespace or prev_ended_sentence or prev_line is None
        mid_sentence = not starts_new_paragraph
        score = 0
        score += 3 if bold else -2
        score += 2 if larger else 0
        score += 2 if starts_margin else 0
        score += 2 if starts_new_paragraph else 0
        score += 2 if ends_colon else 0
        score += 1 if allcaps else 0
        score += 1 if after_whitespace else 0
        score -= 3 if mid_sentence else 0
        if mixed_line and not (starts_margin and ends_colon):
            score -= 2

        item["is_heading"] = score >= threshold

        prev_line = item
        prev_was_break = False


def _merge_multiline_headings(line_items, same_line_tol=3.0, same_line_gap_factor=1.0):
    """Joins a single-line title whose words got a false 'col' split,
    back into one heading. This happens when the inter-word gap in a
    (often larger/bolder) heading font happens to clear min_gap_x even
    though both words sit on the exact same baseline -- e.g. 'KURUMA'
    | 'SÜRESİ' at a ~5pt gap. Detected by: both sides already flagged
    is_heading (this pass never promotes a non-heading line), same
    y-position (same baseline), and a gap small relative to font size.

    Genuine side-by-side column headings (e.g. 'Temizlenebilmesi' /
    'Duvardaki Dokusu' on this document) sit far apart (~53pt here), so
    the gap threshold tells them apart without touching the XY-cut.

    Deliberately does NOT merge across 'row' breaks: on this document,
    every apparent stacked-heading case (e.g. 'TEKNİK BİLGİLER' followed
    by 'Temizlenebilmesi') turned out to be two genuinely distinct
    headings with ordinary spacing, not a wrapped title -- merging
    those was a regression, not a fix. Runs as a pass over the
    already-scored line list, so it can't disturb the XY-cut splitting
    or the scoring logic itself.
    """
    result = []
    i, n = 0, len(line_items)
    while i < n:
        item = line_items[i]
        if item.get("break") or not item.get("is_heading") or item.get("inline_heading_prefix"):
            result.append(item)
            i += 1
            continue

        j = i + 1
        while j < n and line_items[j].get("break") == "col" and j + 1 < n:
            nxt = line_items[j + 1]
            if nxt.get("break") or not nxt.get("is_heading"):
                break
            same_y = abs(nxt["bbox"][1] - item["bbox"][1]) <= same_line_tol
            gap = nxt["bbox"][0] - item["bbox"][2]
            close_enough = same_y and gap <= item.get("dominant_size", 10) * same_line_gap_factor
            if not close_enough:
                break
            item = dict(item)
            item["text"] = item["text"].strip() + " " + nxt["text"].strip()
            item["bbox"] = (
                min(item["bbox"][0], nxt["bbox"][0]),
                min(item["bbox"][1], nxt["bbox"][1]),
                max(item["bbox"][2], nxt["bbox"][2]),
                max(item["bbox"][3], nxt["bbox"][3]),
            )
            j += 2

        result.append(item)
        i = j
    return result


# ---------------------------------------------------------------------------
# 4. Assemble final text: headings, bullets, safe paragraph continuation
# ---------------------------------------------------------------------------

_BULLET_PREFIXES = ("•", "-", "*", "‣", "◦")


def _is_wrap_hyphen(prev_text, next_text):
    """True if `prev_text` ends in a hyphen that's a PDF line-wrap artifact
    splitting one word across two lines (e.g. 'getiril-' / 'melidir' ->
    'getirilmelidir'), as opposed to a real dash or a hyphenated compound.

    Heuristics: the character before the hyphen must be a letter (rules out
    list-style '-' bullets and numeric ranges like '2020-'), and the next
    line must start with a lowercase letter (real compounds/dashes at a
    line end are rare enough, and a following capital usually means a new
    sentence/proper noun rather than a word continuation)."""
    if len(prev_text) < 2 or prev_text[-1] != "-":
        return False
    if not prev_text[-2].isalpha():
        return False
    if not next_text or not next_text[0].isalpha():
        return False
    return next_text[0].islower()


def _join_wrap_hyphen(prev_text, next_text):
    """Merges a hyphen-broken word back together: drops the trailing
    hyphen from `prev_text` and glues `next_text` on with no space."""
    return prev_text[:-1] + next_text


def _assemble_text(line_items):
    """Turns the ordered list of line-dicts/break-markers into final text.

    Headings and bullets are tracked with an explicit `prev_was_break`
    flag rather than a punctuation check, so a heading/bullet can NEVER
    have the next line silently appended onto it (headings/bullets don't
    reliably end in '.', ':' or ';')."""
    out = []
    prev_was_break = True

    for item in line_items:
        if item.get("break"):
            if item["break"] == "col" and out and out[-1] != "":
                out.append("")  # blank line between column/table-cell blocks
            prev_was_break = True
            continue

        text = item["text"].strip()
        if not text:
            continue

        is_bullet = text.startswith(_BULLET_PREFIXES)
        is_heading = item.get("is_heading", False) and not is_bullet

        if is_heading:
            prefix = item.get("inline_heading_prefix")
            if prefix:
                # Bold label glued onto its own paragraph text (e.g.
                # "Ham Sıvalı Yüzey: Öncelikle..."). Emit the label as
                # its own heading, then let the remainder flow as a
                # normal paragraph line -- NOT prev_was_break=True,
                # since that remainder is prose that later lines may
                # still need to continue (same rule as any other line).
                out.append("## " + prefix)
                rest = item.get("inline_heading_rest", "").strip()
                if rest:
                    out.append(rest)
                prev_was_break = False
                continue
            out.append("## " + text)
            prev_was_break = True
            continue

        if is_bullet:
            out.append(text)
            prev_was_break = True
            continue

        if prev_was_break or not out or out[-1] == "":
            out.append(text)
        elif out[-1].endswith(('.', ':', ';', '!', '?')):
            out.append(text)
        elif _is_wrap_hyphen(out[-1], text):
            out[-1] = _join_wrap_hyphen(out[-1], text)
        else:
            out[-1] = out[-1] + " " + text

        prev_was_break = False

    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. Cross-page boilerplate (repeated header/footer) removal
# ---------------------------------------------------------------------------

def _strip_repeated_boilerplate(pages_lines, min_pages_for_filter=4,
                                 freq_ratio=0.8, max_len=120):
    """Removes lines that appear verbatim on most pages of the SAME
    document (running headers/footers, repeated branding/addresses).
    Skipped entirely below `min_pages_for_filter` pages, since on a short
    document an 80% page-overlap threshold could coincidentally strip
    real, non-repeated content."""
    num_pages = len(pages_lines)
    if num_pages < min_pages_for_filter:
        return pages_lines

    page_texts = []
    for lines in pages_lines:
        texts = {
            it["text"].strip()
            for it in lines
            if not it.get("break") and it["text"].strip()
            and len(it["text"].strip()) <= max_len
        }
        page_texts.append(texts)

    counts = Counter()
    for texts in page_texts:
        for t in texts:
            counts[t] += 1

    threshold = max(2, int(num_pages * freq_ratio))
    boilerplate = {t for t, c in counts.items() if c >= threshold}

    cleaned = []
    for lines in pages_lines:
        cleaned.append([
            it for it in lines
            if it.get("break") or it["text"].strip() not in boilerplate
        ])
    return cleaned


# ---------------------------------------------------------------------------
# 6. Public entry point (same signature/return shape as before)
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path, min_gap_x=5, min_gap_y=3, line_tol=2.5,
                      heading_size_ratio=1.1, heading_threshold=6,
                      strip_boilerplate=True):
    """Returns [{"page_number": i, "text": "..."}, ...] with reading-order
    corrected multi-column text, '## ' heading markers, preserved bullet
    lines, and (for 4+ page documents) repeated header/footer lines
    stripped out.

    Tuning:
    - min_gap_x / min_gap_y: minimum empty gap (PDF points) treated as a
      column/row boundary.
    - line_tol: y-distance tolerance (points) for grouping words onto
      the same line.
    - heading_size_ratio: how much larger than body text counts toward
      the "larger than body" heading score.
    - heading_threshold: minimum total score (see _score_and_flag_headings)
      to flag a line as a heading. Raise if too many false-positive
      headings appear; lower if real headings are being missed.
    - strip_boilerplate: set False to keep repeated header/footer lines.
    """
    doc = fitz.open(pdf_path)

    all_sizes = []
    for page in doc:
        all_sizes.extend(_collect_font_sizes(page))
    body_size = _body_font_size(all_sizes)

    pages_lines = []
    page_numbers = []
    for i, page in enumerate(doc):
        spans_style = _all_spans_style(page)
        items = _get_words_with_style(page, spans_style, default_size=body_size)
        if not items:
            continue
        lines = _xy_cut_to_lines(items, min_gap_x=min_gap_x, min_gap_y=min_gap_y,
                                  line_tol=line_tol)
        _score_and_flag_headings(lines, body_size, size_ratio=heading_size_ratio,
                                  threshold=heading_threshold)
        lines = _merge_multiline_headings(lines)
        pages_lines.append(lines)
        page_numbers.append(i + 1)

    doc.close()

    if strip_boilerplate:
        pages_lines = _strip_repeated_boilerplate(pages_lines)

    pages = []
    for page_number, lines in zip(page_numbers, pages_lines):
        text = _assemble_text(lines)
        if text.strip():
            pages.append({"page_number": page_number, "text": text})
    return pages


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset\\TDS_MOMENTOSILAN.pdf"
    for page in extract_pdf_text(path):
        print(f"\n===== Page {page['page_number']} =====")
        print(page["text"])