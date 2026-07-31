"""
Extracts structured paint properties (thinning, drying time, coverage,
airless spray settings, storage, packaging, standards, etc.) from a
Filli Boya-style Technical Data Sheet PDF.

This is intentionally tailored to this document family: it relies on the
'## HEADING' markers that pdf_reading_order2.extract_pdf_text() produces,
and on the fixed set of Turkish section headings that appear in Filli
Boya TDS PDFs (İNCELTME, KURUMA SÜRESİ, SARFİYAT, Havasız (Airless)
Püskürtmede, DEPOLAMA, AMBALAJ, UYARI-1/2, TS .../G tebliğine uygundur).
It will NOT generalize to arbitrary/unrelated PDFs.

Usage:
    python3 extract_properties.py TDS_AQUALUX.pdf
"""

import re
import sys
import json

from pdf_reading_order import extract_pdf_text

#TDS_AQUALUX.pdf
#TDS_EXXENMAT.pdf
PDF_PATH = "dataset/TDS_AQUALUX.pdf"



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
 
 
def _range(pattern, text):
    """Runs `pattern` (must have exactly 2 numeric capture groups) and
    returns (min, max) as numbers, or (None, None) if no match."""
    if not text:
        return None, None
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None, None
    return _num(m.group(1)), _num(m.group(2))
 
 
# ---------------------------------------------------------------------------
# Step 3: extraction
# ---------------------------------------------------------------------------
 
def extract_paint_properties(pdf_path):
    pages = extract_pdf_text(pdf_path)
    sections = {}
    for p in pages:
        sections.update(_sections_from_text(p["text"]))
 
    result = {}
 
    # --- Product name (strip trademark symbols like ®) ------------------
    raw_name = _search(r"##\s*([^\n]+)", pages[0]["text"]) if pages else None
    result["product_name"] = re.sub(r"[®™]", "", raw_name).strip() if raw_name else None
 
    # --- Thinning (İNCELTME) --------------------------------------------
    thinning = _find_section(sections, "İNCELTME") or sections.get("İNCELTME")
    pct = _search(r"%\s*(\d+)", thinning)
    result["thinning_max_percent"] = int(pct) if pct else None
 
    # --- Drying time (KURUMA SÜRESİ) ------------------------------------
    drying = _find_section(sections, "KURUMA", "SÜRE")
    if drying:
        touch_min, touch_max = _range(
            r"Dokunma kuruması:\s*(\d+)\s*[-–]\s*(\d+)\s*dakika", drying)
        result["touch_dry_min_minutes"] = touch_min
        result["touch_dry_max_minutes"] = touch_max
 
        recoat_min, recoat_max = _range(
            r"Katlar arası bekleme süresi:\s*(\d+)\s*[-–]\s*(\d+)\s*saat", drying)
        result["recoat_time_min_hours"] = recoat_min
        result["recoat_time_max_hours"] = recoat_max
 
        cure = _search(r"Son Kuruma:\s*(\d+)\s*saat", drying)
        result["full_cure_hours"] = int(cure) if cure else None
    else:
        for k in ("touch_dry_min_minutes", "touch_dry_max_minutes",
                   "recoat_time_min_hours", "recoat_time_max_hours",
                   "full_cure_hours"):
            result[k] = None
 
    # --- Coverage / spread rate (SARFİYAT) ------------------------------
    coverage = _find_section(sections, "SARFİYAT") or sections.get("SARFİYAT")
    cov_min, cov_max = _range(
        r"(\d+)\s*[-–]\s*(\d+)\s*m\s*2", coverage.replace("\n", " ") if coverage else "")
    result["coverage_min_m2_per_l"] = cov_min
    result["coverage_max_m2_per_l"] = cov_max
 
    # --- Airless spray settings (Havasız (Airless) Püskürtmede) --------
    airless = (_find_section(sections, "havasız")
               or _find_section(sections, "airless")
               or _find_section(sections, "püskürtme"))
    press_min, press_max = _range(r"Basınç:\s*(\d+)\s*[-–]\s*(\d+)\s*bar", airless)
    result["spray_pressure_min_bar"] = press_min
    result["spray_pressure_max_bar"] = press_max
 
    angle = _search(r"Meme açısı:\s*(\d+)\s*°", airless)
    result["spray_nozzle_angle_deg"] = int(angle) if angle else None
 
    nozzle = _search(r"Meme ölçüsü \(inch\):\s*([\d.,]+)", airless)
    result["spray_nozzle_size_inch"] = _num(nozzle) if nozzle else None
 
    # --- Storage (DEPOLAMA) ---------------------------------------------
    storage = _find_section(sections, "DEPOLAMA") or sections.get("DEPOLAMA")
    shelf_life = _search(r"(\d+)\s*yıl", storage)
    result["shelf_life_years"] = int(shelf_life) if shelf_life else None
 
    # --- Packaging (AMBALAJ) --------------------------------------------
    packaging = _find_section(sections, "AMBALAJ") or sections.get("AMBALAJ")
    sizes = re.findall(r"(\d+(?:[.,]\d+)?)\s*Litre", packaging) if packaging else []
    result["packaging_sizes_liters"] = [_num(s) for s in sizes]
 
    # --- Application temperature range (UYARI-1) ------------------------
    uyari1 = sections.get("UYARI-1")
    temp_min, temp_max = _range(r"([+-]?\d+)\s*°C\s*ile\s*([+-]?\d+)\s*°C", uyari1)
    result["application_temp_min_c"] = temp_min
    result["application_temp_max_c"] = temp_max
 
    return result

if __name__ == "__main__":
    path = PDF_PATH
    result = extract_paint_properties(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
