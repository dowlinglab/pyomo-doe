"""Save an exact expanded LaTeX draft for the private evidence-packet audit.

Only controlled, braced input/include forms are expanded. This reads text; it
does not execute TeX or include graphics, external paper text or bibliography
contents in the draft. A preflight inventory does not replace source checking
or explicit review of the literature service's complete citation inventory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from run_fits import save_json

HERE = Path(__file__).resolve().parent
INPUT = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
CITE = re.compile(r"\\(?:[Cc]ite[pt]?|[Aa]utocite|[Pp]arencite|[Tt]extcite|[Ff]ootcite|[Ss]martcite|[Ss]upercite|[Cc]iteauthor|[Cc]iteyear)\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]*)\}")
LIBRARY_CARRIERS = {
    "raue-2009-structural": "6925c25f6cb0061b964f32bc1242089daf43ce271fe57489d97f9e5140ae8160",
    "garnier-2008-direct": "582f0b9c5220871d43ab75ddd96458475e037ecf810ae4ad9e12cc9005424e56",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_tex_regions(text):
    """Preserve offsets while ignoring comments and literal-code environments."""
    masked = list(text)
    regions = [(match.start(), match.end()) for match in re.finditer(r"(?<!\\)%[^\n]*", text)]
    regions.extend((match.start(), match.end()) for match in re.finditer(
        r"\\begin\{(verbatim|lstlisting|minted)\}.*?\\end\{\1\}", text, re.S))
    for start, end in regions:
        masked[start:end] = " "*(end-start)
    return "".join(masked)


def expand_inputs(root, main="main.tex"):
    root = Path(root).resolve()
    sources = {}

    def expand(path, stack):
        path = path.resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            raise ValueError("LaTeX input leaves the manuscript directory")
        if path in stack:
            raise ValueError("Cyclic LaTeX input: "+relative)
        text = path.read_bytes().decode("utf-8")
        sources[relative] = sha(path)
        scan = mask_tex_regions(text)
        matches = list(INPUT.finditer(scan))
        for other in re.finditer(r"\\(?:input|include)\b", scan):
            if not any(match.start() <= other.start() < match.end() for match in matches):
                raise ValueError("Unsupported input form requires explicit review: "+relative)
        pieces, previous = [], 0
        for match in matches:
            child = Path(match[1])
            if not child.suffix:
                child = child.with_suffix(".tex")
            # TeX resolves this project's inputs from the main document's root.
            child = root/child
            pieces.extend([text[previous:match.start()], "\n% Expanded input: "+match[1]+"\n",
                           expand(child, stack+(path,)), "\n% End expanded input\n"])
            previous = match.end()
        pieces.append(text[previous:])
        return "".join(pieces)

    return expand(root/main, ()), sources


def preflight_citations(text):
    scan = mask_tex_regions(text)
    matches = list(CITE.finditer(scan))
    for other in re.finditer(r"\\\w*[Cc]ite\w*", scan):
        if not any(match.start() <= other.start() < match.end() for match in matches):
            raise ValueError("Unsupported citation syntax requires manual inventory review")
    result = []
    for match in matches:
        keys = [key.strip() for key in match[1].split(",")]
        if not all(keys):
            raise ValueError("Empty citation key")
        result.append({"start_char": match.start(), "end_char": match.end(),
                       "text": text[match.start():match.end()], "keys": keys,
                       "line": text.count("\n", 0, match.start())+1})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=HERE/"manuscript")
    parser.add_argument("--output", type=Path, default=HERE/"literature/manuscript_audit")
    args = parser.parse_args()
    text, sources = expand_inputs(args.manuscript)
    citations = preflight_citations(text)
    bib_path = args.manuscript/"bibliography.bib"
    bibliography = bib_path.read_bytes().decode("utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibliography))
    cited = {key for citation in citations for key in citation["keys"]}
    if cited-bib_keys:
        raise RuntimeError("Unmapped bibliography keys: "+str(sorted(cited-bib_keys)))
    for relative, expected in sources.items():
        if sha(args.manuscript/relative) != expected:
            raise RuntimeError("Manuscript source changed during expansion")
    args.output.mkdir(parents=True, exist_ok=True)
    draft_path = args.output/"draft.tex"
    draft_path.write_text(text, encoding="utf-8")
    save_json(args.output/"bibliography_map.json", {key: LIBRARY_CARRIERS.get(key) for key in sorted(cited)})
    save_json(args.output/"preflight.json", {"draft_sha256": sha(draft_path), "source_tex_sha256": sources,
                                            "bibliography_sha256": sha(bib_path), "citations": citations,
                                            "citation_occurrences": len(citations), "cited_works": len(cited),
                                            "unused_bibliography_keys": sorted(bib_keys-cited),
                                            "mapped_library_carriers": sorted(cited & set(LIBRARY_CARRIERS)),
                                            "external_unmapped_carriers": sorted(cited-set(LIBRARY_CARRIERS)),
                                            "chapter_alias_notice": "garnier-2008-direct maps to the stored book carrier; manuscript BibTeX retains the inspected chapter authors, title, pages and chapter DOI, rather than replacing them with the book-editor export.",
                                            "producer_sha256": sha(HERE/"prepare_manuscript_audit.py"),
                                            "status": "prepared_for_private_exact_draft_packet_review",
                                            "notice": "Text/offset and bibliography preflight only. Review all prose references and the service inventory explicitly, bind claims, inspect source context, and report fine-grounded versus externally inspected/unavailable evidence separately. No scientific acceptance is implied."})
    print(json.dumps({"citation_occurrences": len(citations), "cited_works": len(cited),
                      "source_tex_files": len(sources), "draft_sha256": sha(draft_path)}))


if __name__ == "__main__":
    main()
