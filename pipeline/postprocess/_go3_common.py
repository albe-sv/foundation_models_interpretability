"""Shared GO3 / GWAS Catalog helpers for gwas_similarity.py and gene_matrix.py.

go3 is a local Rust/PyO3 package (not on PyPI); this module only assumes it is
importable in the running environment (it already is inside scgpt_env).

go-basic.obo and goa_human.gaf are the monorepo's own reference copies
(GO3/go-basic.obo, data/goa_human.gaf at the repo root); interp_repro_mixed
expects them symlinked under data/go3_refs/ rather than duplicated, since
data/ is gitignored here.
"""

from pathlib import Path

import gseapy
import pandas as pd

import go3

REFS_DIR = Path(__file__).resolve().parents[2] / "data" / "go3_refs"
OBO_PATH = REFS_DIR / "go-basic.obo"
GAF_PATH = REFS_DIR / "goa_human.gaf"

GWAS_LIBRARY = "GWAS_Catalog_2025"
GWAS_TRAITS = {"parkinson": "Parkinsons Disease",
               "alzheimer": "Alzheimers Disease",
               "cad": "Coronary Artery Disease"}
ASPECT_TO_GO_SOURCE = {"P": "GO:BP", "C": "GO:CC", "F": "GO:MF"}


# Fetch one GWAS Catalog trait's gene set, caching it to `cache` on first use
def fetch_gwas(trait: str, cache: Path) -> list:
    if cache.exists():
        return pd.read_csv(cache)["gene"].astype(str).tolist()
    lib = gseapy.get_library(GWAS_LIBRARY)
    # Trait names in the library are free text, so match case/whitespace-insensitively
    match = [g for t, g in lib.items() if t.strip().lower() == trait.lower()]
    if not match:
        raise ValueError(f"trait {trait!r} not found in {GWAS_LIBRARY}")
    genes = sorted(dict.fromkeys(g for gs in match for g in gs))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": genes}).to_csv(cache, index=False)
    print(f"GWAS '{trait}': {len(genes)} genes -> {cache}")
    return genes

# Fetch (or reuse the cache of) every GWAS disease
def fetch_all_gwas(out_dir: Path) -> dict:
    return {key: fetch_gwas(trait, out_dir / f"{key}_genes.csv") for key, trait in GWAS_TRAITS.items()}

# This function gets all Gene symbols at least one GO annotation
def gaf_symbols_by_go_source() -> dict:
    gaf = pd.read_csv(GAF_PATH, sep="\t", comment="!", header=None, usecols=[2, 8],
                      names=["symbol", "aspect"], dtype=str, low_memory=False)
    return {go_source: set(gaf.loc[gaf["aspect"] == aspect, "symbol"])
            for aspect, go_source in ASPECT_TO_GO_SOURCE.items()}

# Load the GO ontology and build the annotation counter used by every go3
def init_go3(threads: int = 16):
    go3.set_num_threads(threads)
    go3.load_go_terms(str(OBO_PATH))
    return go3.build_term_counter(go3.load_gaf(str(GAF_PATH)))
