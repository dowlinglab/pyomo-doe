"""An exact draft cannot silently omit nested text or expand outside files."""
import pytest

from prepare_manuscript_audit import expand_inputs, preflight_citations


def test_nested_expansion_retains_citation_offsets_and_ignores_literal_commands(tmp_path):
    (tmp_path/"main.tex").write_text("Before\\input{a}After\n% \\input{absent}\n")
    (tmp_path/"a.tex").write_text("Claim \\citet[p.~4]{one}.\\input{b}\n")
    (tmp_path/"b.tex").write_text("More \\citep{two,three}.\\begin{verbatim}\\cite{ignored}\\end{verbatim}")
    text, sources = expand_inputs(tmp_path)
    citations = preflight_citations(text)
    assert set(sources) == {"main.tex", "a.tex", "b.tex"}
    assert [item["keys"] for item in citations] == [["one"], ["two", "three"]]
    assert all(text[item["start_char"]:item["end_char"]] == item["text"] for item in citations)
    assert text.index("Before") < text.index("Claim") < text.index("More") < text.index("After")


def test_cycles_and_outside_inputs_fail_before_producing_an_incomplete_draft(tmp_path):
    (tmp_path/"main.tex").write_text("\\input{a}")
    (tmp_path/"a.tex").write_text("\\input{main}")
    with pytest.raises(ValueError, match="Cyclic"):
        expand_inputs(tmp_path)
    (tmp_path/"a.tex").write_text("\\input{../outside}")
    with pytest.raises(ValueError, match="leaves the manuscript"):
        expand_inputs(tmp_path)


def test_unrecognized_citation_syntax_cannot_be_silently_waived():
    with pytest.raises(ValueError, match="manual inventory"):
        preflight_citations("Unrecognized \\citecustom{key}")
