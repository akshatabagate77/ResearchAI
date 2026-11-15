
import json
import pytest

import src.research_summarizer as summarizer


def sample_chunks():

    return [
        {
            "document_name": "AI_Paper.pdf",
            "page_number": 1,
            "text": "Machine learning helps computers learn from data."
        },
        {
            "document_name": "AI_Paper.pdf",
            "page_number": 2,
            "text": "Deep learning uses neural networks."
        }
    ]


def test_create_document_id_is_consistent():

    chunks = sample_chunks()

    first_id = summarizer.create_document_id(
        "AI_Paper.pdf", chunks
    )

    second_id = summarizer.create_document_id(
        "AI_Paper.pdf", chunks
    )

    assert first_id == second_id


def test_create_document_id_changes_with_content():

    chunks = sample_chunks()

    first_id = summarizer.create_document_id(
        "AI_Paper.pdf", chunks
    )

    chunks[0]["text"] = "Different research content."

    second_id = summarizer.create_document_id(
        "AI_Paper.pdf", chunks
    )

    assert first_id != second_id


def test_save_and_load_checkpoint(tmp_path, monkeypatch):

    monkeypatch.setattr(
        summarizer,
        "CHECKPOINT_DIR",
        str(tmp_path)
    )

    checkpoint = {
        "completed_batches": {
            "0": "First batch summary"
        },
        "final_summary": "Final research summary"
    }

    document_id = "sample123"

    summarizer.save_checkpoint(document_id, checkpoint)

    loaded = summarizer.load_checkpoint(document_id)

    assert loaded == checkpoint


def test_load_checkpoint_returns_empty_for_missing_file(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(
        summarizer,
        "CHECKPOINT_DIR",
        str(tmp_path)
    )

    result = summarizer.load_checkpoint("missing123")

    assert result["completed_batches"] == {}
    assert result["final_summary"] is None


def test_analyse_batch_uses_generated_answer(monkeypatch):

    monkeypatch.setattr(
        summarizer,
        "generate_answer",
        lambda prompt: "Research objective: Study machine learning."
    )

    result = summarizer.analyse_batch(
        "AI_Paper.pdf",
        sample_chunks(),
        1
    )

    assert "Research objective" in result


def test_analyse_batch_rejects_empty_response(monkeypatch):

    monkeypatch.setattr(
        summarizer,
        "generate_answer",
        lambda prompt: ""
    )

    with pytest.raises(ValueError):

        summarizer.analyse_batch(
            "AI_Paper.pdf",
            sample_chunks(),
            1
        )


def test_create_final_summary(monkeypatch):

    monkeypatch.setattr(
        summarizer,
        "generate_answer",
        lambda prompt: "Final structured research summary."
    )

    result = summarizer.create_final_summary(
        "AI_Paper.pdf",
        ["First section analysis", "Second section analysis"]
    )

    assert result == "Final structured research summary."


def test_create_final_summary_rejects_empty_sections():

    with pytest.raises(ValueError):

        summarizer.create_final_summary(
            "AI_Paper.pdf",
            []
        )


def test_summarize_research_papers_validates_empty_chunks():

    with pytest.raises(ValueError):

        summarizer.summarize_research_papers([])


def test_summarize_research_papers_validates_batch_size():

    with pytest.raises(ValueError):

        summarizer.summarize_research_papers(
            sample_chunks(),
            batch_size=0
        )


def test_summarize_research_papers_rejects_invalid_chunk():

    with pytest.raises(ValueError):

        summarizer.summarize_research_papers(
            [{"document_name": "AI_Paper.pdf", "text": ""}]
        )


def test_summarize_research_papers_rejects_invalid_force_refresh():

    with pytest.raises(ValueError):

        summarizer.summarize_research_papers(
            sample_chunks(),
            force_refresh="yes"
        )


def test_summarize_research_papers_processes_document(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(
        summarizer,
        "CHECKPOINT_DIR",
        str(tmp_path)
    )

    monkeypatch.setattr(
        summarizer,
        "analyse_batch",
        lambda *args: "Section analysis"
    )

    monkeypatch.setattr(
        summarizer,
        "create_final_summary",
        lambda *args: "Final research summary"
    )

    result = summarizer.summarize_research_papers(
        sample_chunks(),
        batch_size=2,
        force_refresh=True
    )

    assert "AI_Paper.pdf" in result
    assert result["AI_Paper.pdf"] == "Final research summary"


def test_summarize_reuses_cached_final_summary(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(
        summarizer,
        "CHECKPOINT_DIR",
        str(tmp_path)
    )

    chunks = sample_chunks()

    document_id = summarizer.create_document_id(
        "AI_Paper.pdf",
        chunks,
        batch_size=8
    )

    checkpoint = {
        "completed_batches": {},
        "final_summary": "Previously saved summary"
    }

    summarizer.save_checkpoint(document_id, checkpoint)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("AI generation should not be called.")

    monkeypatch.setattr(
        summarizer,
        "analyse_batch",
        unexpected_call
    )

    result = summarizer.summarize_research_papers(chunks)

    assert result["AI_Paper.pdf"] == "Previously saved summary"



# Test 15: Handle invalid JSON checkpoint
def test_load_checkpoint_handles_invalid_json(tmp_path, monkeypatch):

    monkeypatch.setattr(
        summarizer, "CHECKPOINT_DIR", str(tmp_path)
    )

    checkpoint_file = tmp_path / "invalid.json"
    checkpoint_file.write_text(
        "{invalid json",
        encoding="utf-8"
    )

    result = summarizer.load_checkpoint("invalid")

    assert result == {
        "completed_batches": {},
        "final_summary": None
    }


# Test 16: Reject invalid checkpoint data
@pytest.mark.parametrize(
    "checkpoint",
    [
        [],
        {"completed_batches": []},
        {"completed_batches": {}, "final_summary": 123}
    ]
)
def test_load_checkpoint_rejects_invalid_data(
    tmp_path, monkeypatch, checkpoint
):

    monkeypatch.setattr(
        summarizer, "CHECKPOINT_DIR", str(tmp_path)
    )

    checkpoint_file = tmp_path / "invalid.json"
    checkpoint_file.write_text(
        json.dumps(checkpoint),
        encoding="utf-8"
    )

    result = summarizer.load_checkpoint("invalid")

    assert result == {
        "completed_batches": {},
        "final_summary": None
    }


# Test 17: Clean temporary file after save failure
def test_save_checkpoint_cleans_temp_file_after_failure(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(
        summarizer, "CHECKPOINT_DIR", str(tmp_path)
    )

    def fake_replace(source, destination):
        raise OSError("Simulated replace failure")

    monkeypatch.setattr(
        summarizer.os, "replace", fake_replace
    )

    with pytest.raises(OSError, match="Simulated replace failure"):
        summarizer.save_checkpoint(
            "sample",
            {"completed_batches": {}, "final_summary": None}
        )

    temporary_file = tmp_path / "sample.json.tmp"

    assert not temporary_file.exists()


# Test 18: Handle temporary-file removal failure
def test_save_checkpoint_handles_cleanup_error(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(
        summarizer, "CHECKPOINT_DIR", str(tmp_path)
    )

    def fake_replace(source, destination):
        raise OSError("Simulated replace failure")

    def fake_remove(path):
        raise OSError("Simulated remove failure")

    monkeypatch.setattr(
        summarizer.os, "replace", fake_replace
    )
    monkeypatch.setattr(
        summarizer.os, "remove", fake_remove
    )

    with pytest.raises(OSError, match="Simulated replace failure"):
        summarizer.save_checkpoint(
            "sample",
            {"completed_batches": {}, "final_summary": None}
        )


# Test 19: Reject empty final summary response
def test_create_final_summary_rejects_empty_response(monkeypatch):

    monkeypatch.setattr(
        summarizer,
        "generate_answer",
        lambda prompt: ""
    )

    with pytest.raises(ValueError, match="empty response"):
        summarizer.create_final_summary(
            "AI_Paper.pdf",
            ["Section analysis"]
        )


# Test 20: Reject a chunk that is not a dictionary
def test_summarize_rejects_non_dictionary_chunk():

    with pytest.raises(ValueError, match="must be a dictionary"):
        summarizer.summarize_research_papers(
            ["This is not a dictionary"]
        )


# Test 21: Reuse a completed batch
def test_summarize_reuses_completed_batch(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(
        summarizer, "CHECKPOINT_DIR", str(tmp_path)
    )

    chunks = sample_chunks()

    document_id = summarizer.create_document_id(
        "AI_Paper.pdf",
        chunks,
        batch_size=8
    )

    summarizer.save_checkpoint(
        document_id,
        {
            "completed_batches": {
                "0": "Previously completed section"
            },
            "final_summary": None
        }
    )

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError(
            "Completed batch should not be analysed again."
        )

    monkeypatch.setattr(
        summarizer, "analyse_batch", unexpected_analysis
    )

    monkeypatch.setattr(
        summarizer,
        "create_final_summary",
        lambda *args: "Final summary from saved batch"
    )

    result = summarizer.summarize_research_papers(chunks)

    assert result["AI_Paper.pdf"] == "Final summary from saved batch"


# Test 22: Detect a missing completed batch
def test_summarize_detects_missing_completed_batch(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(
        summarizer, "CHECKPOINT_DIR", str(tmp_path)
    )

    monkeypatch.setattr(
        summarizer,
        "analyse_batch",
        lambda *args: ""
    )

    with pytest.raises(RuntimeError, match="Missing completed batch"):
        summarizer.summarize_research_papers(
            sample_chunks(),
            batch_size=2,
            force_refresh=True
        )
