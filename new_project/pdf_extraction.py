import re
import sys
import json
from pdf_reading_order import extract_pdf_text


#TDS_AQUALUX.pdf
#TDS_EXXENMAT.pdf
#TDS_WOODMAXX_WOODSTAIN.pdf
#TDS_MOMENTOPLASTIX.pdf
#TDS_AQUSTO_SILAN.pdf
PDF_PATH = "dataset/TDS_MOMENTOSILAN.pdf"



# ---------------------------------------------------------------------------
# Step 1: split the assembled text of a page into {heading: body_text}
# ---------------------------------------------------------------------------
 
def _sections_from_text(text):
    """Splits '## HEADING\nbody...' formatted text into a dict of
    {heading: body}. A heading with an empty body (its content actually
    landed under the *next* heading -- happens when a heading wraps onto
    two physical lines, e.g. 'Havasız (Airless)' / 'Püskürtmede') gets
    folded into that next heading's title."""
    sections = {}
    order = []
    current = None
    buf = []
 
    for line in text.split("\n"):
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            order.append(current)
            buf = []
        else:
            if current is not None:
                buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
 
    merged = {}
    i = 0
    while i < len(order):
        head = order[i]
        body = sections[head]
        if not body.strip() and i + 1 < len(order):
            nxt = order[i + 1]
            merged[(head + " " + nxt).strip()] = sections[nxt]
            i += 2
            continue
        merged[head] = body
        i += 1
    return merged
 
 
# ---------------------------------------------------------------------------
# Step 2: small regex/number helpers
# ---------------------------------------------------------------------------
 
def _search(pattern, text, group=1, flags=re.IGNORECASE | re.DOTALL):
    if not text:
        return None
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None
 
 
def _find_section(sections, *keywords):
    """Body text of the first section whose heading contains ALL keywords
    (case-insensitive substring match)."""
    for heading, body in sections.items():
        h = heading.lower()
        if all(k.lower() in h for k in keywords):
            return body
    return None
 
 
def _num(s):
    """'0,019' / '0.019' / '50' -> float/int."""
    if s is None:
        return None
    s = s.replace(",", ".").strip()
    val = float(s)
    return int(val) if val.is_integer() else val
 
 



def _range_unit(pattern, text):
    """Runs `pattern` (must have exactly 3 capture groups: min, max, unit)
    and returns (min, max, unit) -- the unit is read directly from the
    document text, not assumed. Returns (None, None, None) if no match."""
    if not text:
        return None, None, None
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None, None, None
    return _num(m.group(1)), _num(m.group(2)), m.group(3).strip()


def _value_unit(pattern, text):
    """Runs `pattern` (must have exactly 2 capture groups: value, unit)
    and returns (value, unit) -- the unit is read directly from the
    document text. Returns (None, None) if no match."""
    if not text:
        return None, None
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None, None
    return _num(m.group(1)), m.group(2).strip()


def _percentage_interval(text):
    """Reads a thinning percentage out of `text`. Returns (value, "%"):
      - value is a single number, e.g. 5, when only one percentage is
        given ("%5 inceltilerek").
      - value is a (lo, hi) tuple, e.g. (20, 25), when the text gives a
        range ("maksimum %20-25 inceltilerek"). This is deliberately
        NOT labelled (lo, hi) as "min"/"max": a phrase like "maksimum
        %20-25" means the *maximum* thinning allowed falls somewhere in
        that 20-25 band -- 20 is not "the minimum thinning" and 25 is
        not "the maximum thinning", they're just the two ends of one
        interval, so both numbers are returned together as-is.
      - value is 0 when `text` is non-empty but no percentage is found
        at all ("Kullanıma hazırdır" / ready-to-use / must-not-be-
        thinned wording, phrased differently across datasheets).
      - value is None (with unit None) when `text` itself is missing.

    The '%' can come BEFORE the number ("%5") just as often as after
    ("5%"), so both orders are tried, for both the single-value and the
    range case."""
    if not text:
        return None, None

    # A '%' must actually be part of the match (at the start or the
    # end) -- otherwise an unrelated "N-M" pair in the text (e.g. a
    # drying-time range like "4-6 saat") could be mistaken for a
    # thinning ratio just because it's nearby.
    range_match = re.search(
        r"%\s*(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*%?"
        r"|(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*%", text)
    if range_match:
        lo = range_match.group(1) or range_match.group(3)
        hi = range_match.group(2) or range_match.group(4)
        return (_num(lo), _num(hi)), "%"

    single_match = re.search(r"%\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*%", text)
    if single_match:
        value = _num(single_match.group(1) or single_match.group(2))
        return value, "%"

    # Text is there (we know this is a thinning-related passage), but
    # no number anywhere in it -- ready to use / must not be thinned.
    return 0, "%"


def _inceltme_firca_rulo(inceltme_text):
    """İnceltme (thinning) ratio for brush/roller application (fırça ve
    rulo uygulaması). This lives in its own dedicated "İNCELTME"
    section, so unlike the airless case below the whole section body
    IS the thinning passage -- no need to anchor on the word
    "inceltme" first (datasheets usually phrase this with verb forms
    like "inceltilerek"/"inceltilir" rather than the noun itself, so
    that anchor wouldn't reliably fire here anyway). Returns
    (value, unit) -- see _percentage_interval for the shape of value.
    (None, None) if the section itself wasn't found."""
    return _percentage_interval(inceltme_text)


def _inceltme_havasiz(airless_text):
    """İnceltme (thinning) ratio for airless/spray application (havasız
    püskürtme) -- lives inside the airless/spray section, right
    alongside basınç/meme bilgileri. Unlike the brush/roller section,
    this section has other numbers in it (basınç, meme ölçüsü), so we
    anchor on the literal "inceltme" label first and only look a short
    distance past it, to avoid picking up an unrelated number. Returns
    (value, unit) -- see _percentage_interval for the shape of value.
    (None, None) if the section text doesn't even mention "inceltme"."""
    if not airless_text:
        return None, None

    mention = re.search(r"[İIiı]nceltme", airless_text)
    if not mention:
        return None, None

    # Only look a short distance past the "inceltme" mention, so an
    # unrelated number elsewhere in the (possibly long) airless section
    # doesn't get mistaken for the thinning ratio.
    window = airless_text[mention.end(): mention.end() + 200]
    return _percentage_interval(window)
 
 
# ---------------------------------------------------------------------------
# Step 3: extraction
# ---------------------------------------------------------------------------
 
def extract_paint_properties(pages):
    sections = {}
    for p in pages:
        sections.update(_sections_from_text(p["text"]))
 
    result = {}
 
    # --- Product name (strip trademark symbols like ®) ------------------
    raw_name = _search(r"##\s*([^\n]+)", pages[0]["text"]) if pages else None
    result["ürün_adı"] = re.sub(r"[®™]", "", raw_name).strip() if raw_name else None

    intro_match = re.search(r"^##[^\n]*\n(.*?)(?=\n##\s|\Z)",
                             pages[0]["text"] if pages else "", re.DOTALL)
    intro_paragraph = intro_match.group(1).strip() if intro_match else ""

    # --- Su bazlı mı? (water-based) --------------------------------------
    result["su_bazlı"] = bool(re.search(r"su\s*bazlı", intro_paragraph, re.IGNORECASE))


    result["voc_uyumlu"] = bool(re.search(r"VOC\s*Uyumlu", intro_paragraph, re.IGNORECASE))

    kullanım_match = re.search(r"(iç\s*cephe|dış\s*cephe)", intro_paragraph, re.IGNORECASE)
    result["kullanım_alanı"] = re.sub(r"\s+", " ", kullanım_match.group(1)).strip().lower() if kullanım_match else None

    # --- Doku: parlak mı mat mı? ------------------------------------------
    # Ordered most-specific-first so "lüks parlak" / "yarı mat" etc. win
    # over the bare "parlak"/"mat" alternative.
    doku_match = re.search(
        r"(lüks\s*parlak|yarı\s*parlak|tam\s*parlak|parlak|yarı\s*mat|tam\s*mat|mat)",
        intro_paragraph, re.IGNORECASE)
    result["doku"] = doku_match.group(1).strip() if doku_match else None
 
    # --- Drying time (KURUMA SÜRESİ) ------------------------------------
    drying = _find_section(sections, "KURUMA", "SÜRE")
    if drying:
        touch_min, touch_max, touch_unit = _range_unit(
            r"Dokunma kuruması:\s*(\d+)\s*[-–]\s*(\d+)\s*(dakika|saat|gün)", drying)
        result["dokunma_kuruması_min"] = touch_min
        result["dokunma_kuruması_max"] = touch_max
        result["dokunma_kuruması_birimi"] = touch_unit
 
        recoat_min, recoat_max, recoat_unit = _range_unit(
            r"Katlar arası bekleme süresi:\s*(\d+)\s*[-–]\s*(\d+)\s*(dakika|saat|gün)", drying)
        result["katlar_arası_bekleme_min"] = recoat_min
        result["katlar_arası_bekleme_max"] = recoat_max
        result["katlar_arası_bekleme_birimi"] = recoat_unit
 
        cure_val, cure_unit = _value_unit(
            r"(?:Son|Sert)\s*Kuruma(?:sı)?\s*:\s*(\d+)\s*(dakika|saat|gün)", drying)
        result["son_kuruma"] = cure_val
        result["son_kuruma_birimi"] = cure_unit

    else:
        for k in ("dokunma_kuruması_min", "dokunma_kuruması_max", "dokunma_kuruması_birimi",
                   "katlar_arası_bekleme_min", "katlar_arası_bekleme_max", "katlar_arası_bekleme_birimi",
                   "son_kuruma", "son_kuruma_birimi"):
            result[k] = None
 
    # --- Coverage / spread rate (SARFİYAT) ------------------------------
    coverage = _find_section(sections, "SARFİYAT") or sections.get("SARFİYAT")
    cov_min, cov_max, cov_unit = _range_unit(
        r"(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*(m\s?2|m²)",
        coverage.replace("\n", " ") if coverage else "")
    result["sarfiyat_min"] = cov_min
    result["sarfiyat_max"] = cov_max
    # normalize whitespace/superscript variants of the captured unit (e.g. "m 2", "m2") to "m²"
    result["sarfiyat_birimi"] = re.sub(r"m\s?2", "m²", cov_unit) + "/Litre" if cov_unit else None
 
    # --- Airless spray settings (Havasız (Airless) Püskürtmede) --------
    airless = (_find_section(sections, "havasız")
               or _find_section(sections, "airless")
               or _find_section(sections, "püskürtme"))
    #press_min, press_max, press_unit = _range_unit(r"Basınç:\s*(\d+)\s*[-–]\s*(\d+)\s*(bar|psi)", airless)
    #result["havasız_püskürtme_basıncı_min"] = press_min
    #result["havasız_püskürtme_basıncı_max"] = press_max
    #result["havasız_püskürtme_basıncı_birimi"] = press_unit

    inceltme_havasiz, inceltme_havasiz_birimi = _inceltme_havasiz(airless)
    result["inceltme_havasız_püskürtme"] = inceltme_havasiz
    result["inceltme_havasız_püskürtme_birimi"] = inceltme_havasiz_birimi

    inceltme_section = _find_section(sections, "ncelt")
    inceltme_firca_rulo, inceltme_firca_rulo_birimi = _inceltme_firca_rulo(inceltme_section)
    result["inceltme_fırça_rulo"] = inceltme_firca_rulo
    result["inceltme_fırça_rulo_birimi"] = inceltme_firca_rulo_birimi

    # --- Storage (DEPOLAMA) ---------------------------------------------
    storage = _find_section(sections, "DEPOLAMA") or sections.get("DEPOLAMA")
    shelf_val, shelf_unit = _value_unit(r"(\d+)\s*(yıl|ay|gün)", storage)
    result["depolama_süresi"] = shelf_val
    result["depolama_süresi_birimi"] = shelf_unit
 
    # --- Packaging (AMBALAJ) --------------------------------------------
    packaging = _find_section(sections, "AMBALAJ") or sections.get("AMBALAJ")
    sizes = re.findall(r"(\d+(?:[.,]\d+)?)\s*(Litre|litre|L)\b", packaging) if packaging else []
    result["ambalaj_boyutları"] = [_num(s[0]) for s in sizes]
    result["ambalaj_boyutları_birimi"] = sizes[0][1] if sizes else None
 

    return result

if __name__ == "__main__":
    path = PDF_PATH
    pages = extract_pdf_text(path)
    result = extract_paint_properties(pages)
    print(json.dumps(result, ensure_ascii=False, indent=2))