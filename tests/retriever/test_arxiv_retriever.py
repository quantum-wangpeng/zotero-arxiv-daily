"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]



def test_arxiv_retriever_falls_back_to_per_paper_on_batch_406(
    config, mock_feedparser, monkeypatch
):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    fake_results_by_id = {}
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results_by_id[pid] = SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        )

    warnings: list[str] = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            ids = list(search.id_list)
            if len(ids) > 1:
                raise arxiv_retriever.arxiv.HTTPError(
                    "https://export.arxiv.org/api/query", 0, 406
                )
            return iter([fake_results_by_id[ids[0]]])

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(
        arxiv_retriever,
        "logger",
        SimpleNamespace(warning=warnings.append, info=lambda _: None),
    )
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert any("Falling back to per-paper requests" in warning for warning in warnings)


def test_arxiv_retriever_skips_single_paper_that_still_returns_406(
    config, mock_feedparser, monkeypatch
):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    fake_results_by_id = {}
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results_by_id[pid] = SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        )

    skipped_id = next(iter(fake_results_by_id))
    warnings: list[str] = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            ids = list(search.id_list)
            if len(ids) > 1:
                raise arxiv_retriever.arxiv.HTTPError(
                    "https://export.arxiv.org/api/query", 0, 406
                )
            if ids[0] == skipped_id:
                raise arxiv_retriever.arxiv.HTTPError(
                    "https://export.arxiv.org/api/query", 0, 406
                )
            return iter([fake_results_by_id[ids[0]]])

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(
        arxiv_retriever,
        "logger",
        SimpleNamespace(warning=warnings.append, info=lambda _: None),
    )
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries) - 1
    assert skipped_id not in {
        paper.url.removeprefix("https://arxiv.org/abs/")
        for paper in papers
    }
    assert any(
        f"Skipping arXiv paper {skipped_id}" in warning
        for warning in warnings
    )
