import pytest

from tenerife_rag.citations import cited_labels, has_citation, strip_citations

FORMS = [
    ("Benijo [S1].", {"S1"}),
    ("Benijo [S1][S2].", {"S1", "S2"}),
    ("Benijo [S1, S2].", {"S1", "S2"}),
    ("Benijo [S1; S2].", {"S1", "S2"}),
    ("Benijo [ S1 ,S12 ].", {"S1", "S12"}),
]


@pytest.mark.parametrize(("text", "labels"), FORMS)
def test_cited_labels_recognises_every_form(text, labels):
    assert cited_labels(text) == labels


@pytest.mark.parametrize(("text", "labels"), FORMS)
def test_has_citation_recognises_every_form(text, labels):
    assert has_citation(text)


@pytest.mark.parametrize(("text", "labels"), FORMS)
def test_strip_citations_removes_every_form(text, labels):
    assert strip_citations(text) == "Benijo."


@pytest.mark.parametrize("text", ["Sin citas.", "Una lista [1, 2].", "Sección [S].", "Texto [Sx]."])
def test_text_without_labels_is_not_a_citation(text):
    assert not has_citation(text)
    assert cited_labels(text) == set()
    assert strip_citations(text) == text
