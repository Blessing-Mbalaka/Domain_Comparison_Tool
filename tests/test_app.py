import json

import app.main as main


def test_seed_embedding_jobs_and_persistence(tmp_path):
    main.DATA = tmp_path / "data"
    main.UPLOADS = main.DATA / "uploads"
    main.DB_PATH = main.DATA / "test.sqlite3"
    with main.connect() as db:
        main.seed(db)
        assert db.execute("SELECT COUNT(*) FROM domains").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM categories").fetchone()[0] == 17
        assert db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 18
        assert db.execute("SELECT COUNT(*) FROM legacy_scores").fetchone()[0] == 576
        assert db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 1224
        assert db.execute("SELECT COUNT(*) FROM papers WHERE source_path='synthetic demo presence corpus'").fetchone()[0] == 6
        assert db.execute("SELECT COUNT(*) FROM relationships r JOIN papers p ON p.id=r.paper_id WHERE p.source_path='synthetic demo presence corpus' AND r.exact_presence=1").fetchone()[0] > 0
        domain_id = db.execute("SELECT id FROM domains WHERE name='Research Methodology / Tools'").fetchone()[0]
        stamp = main.now()
        cid = db.execute("INSERT INTO categories(domain_id,name,description,synonyms,position,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", (domain_id, "Interview Analysis", "Analyze interview evidence", json.dumps(["qualitative interviews"]), 9, stamp, stamp)).lastrowid
        pid = db.execute("INSERT INTO papers(title,text_content,status,selected,created_at,updated_at) VALUES(?,?,?,?,?,?)", ("Upload test", "Risk management uses qualitative interviews and project controls.", "processing", 1, stamp, stamp)).lastrowid
        main.prepare_paper(db, pid)
        db.commit()
        rel = db.execute("SELECT * FROM relationships WHERE paper_id=? AND category_id=? AND model_id='mpnet-primary'", (pid, cid)).fetchone()
        assert rel is not None
        assert json.loads(rel["top_terms"])
        assert json.loads(rel["evidence"])
        assert rel["exact_presence"] == 0
        assert rel["concept_presence"] == 1
        assert rel["presence_reason"].startswith("synonym:")
        classified = main.update_paper(pid, main.PaperIn(title="Upload test", text_content="Risk management uses qualitative interviews and project controls.", stage_id=cid))
        assert classified["recalculation"] is True
        assert main.process_one_job() is True
        classified_rel = db.execute("SELECT concept_presence,presence_reason FROM relationships WHERE paper_id=? AND category_id=? AND model_id='mpnet-primary'", (pid, cid)).fetchone()
        assert classified_rel["concept_presence"] == 1
        assert classified_rel["presence_reason"] == "classified stage"
        edited_text = "Corrected spelling for qualitative interview analysis and project risk controls."
        edited = main.update_paper(pid, main.PaperIn(title="Corrected upload title", text_content=edited_text))
        assert edited["recalculation"] is True
        assert edited["job_id"] is not None
        assert main.process_one_job() is True
        refreshed = db.execute("SELECT title,text_content,status FROM papers WHERE id=?", (pid,)).fetchone()
        assert refreshed["title"] == "Corrected upload title"
        assert refreshed["text_content"] == edited_text
        assert refreshed["status"] == "ready"
        assert db.execute("SELECT COUNT(*) FROM relationships WHERE paper_id=?", (pid,)).fetchone()[0] > 0
        title_only = main.update_paper(pid, main.PaperIn(title="Corrected upload heading", text_content=edited_text))
        assert title_only["recalculation"] is False
        assert title_only["job_id"] is None
        pm_id = db.execute("SELECT id FROM domains WHERE name='Project Management'").fetchone()[0]
        research_stage = db.execute("SELECT id FROM categories WHERE domain_id=? ORDER BY id LIMIT 1", (domain_id,)).fetchone()[0]
        pm_stage = db.execute("SELECT id FROM categories WHERE domain_id=? ORDER BY id LIMIT 1", (pm_id,)).fetchone()[0]
        comparative = main.update_paper(pid, main.PaperIn(title="Corrected upload heading", text_content=edited_text, stage_id=cid, comparison_stage_id=pm_stage))
        assert comparative["recalculation"] is True
        assert main.process_one_job() is True
        assert db.execute("SELECT category_id FROM paper_comparisons WHERE paper_id=?", (pid,)).fetchone()[0] == pm_stage
        bulk_pid = db.execute("INSERT INTO papers(title,text_content,status,selected,created_at,updated_at) VALUES(?,?,?,?,?,?)", ("Bulk filename title", "A bulk paper about research design and delivery controls.", "queued", 1, stamp, stamp)).lastrowid
        db.execute("INSERT INTO paper_categories(paper_id,category_id,created_at) VALUES(?,?,?)", (bulk_pid, research_stage, stamp))
        db.execute("INSERT INTO paper_comparisons(paper_id,category_id,created_at) VALUES(?,?,?)", (bulk_pid, pm_stage, stamp))
        db.commit()
        bulk_job = main.queue_job(db, "paper.bulk_process", {"paper_ids": [bulk_pid]})
        assert main.process_one_job() is True
        assert db.execute("SELECT status FROM papers WHERE id=?", (bulk_pid,)).fetchone()[0] == "ready"
        assert db.execute("SELECT status FROM jobs WHERE id=?", (bulk_job,)).fetchone()[0] == "completed"
        db.commit()
        bi = main.stage_alignment(pm_id, domain_id, model_ids="mpnet-primary", metric="alignment")
        assert bi["mode"] == "bi"
        assert len(bi["columns"]) == 12
        assert len(bi["rows"]) == 6
        assert len(bi["cells"]) == 72
        concept_bi = main.stage_alignment(pm_id, domain_id, model_ids="mpnet-primary", metric="concept_presence")
        assert all(0 <= cell["concept_presence"] <= 1 for cell in concept_bi["cells"])
        assert all(0 <= cell["x_present_documents"] <= cell["x_total_documents"] for cell in concept_bi["cells"])
        assert all(0 <= cell["y_present_documents"] <= cell["y_total_documents"] for cell in concept_bi["cells"])
        assert concept_bi["cells"][0]["x_total_documents"] > 0
        analytics = main.analytics_summary()
        assert len(analytics["top_words"]) <= 100
        assert len(analytics["document_words"]) == analytics["documents"]
        assert all(len(document["top_words"]) <= 100 for document in analytics["document_words"])
        assert all(document["total_words"] >= document["unique_words"] for document in analytics["document_words"])
        assert len(analytics["model_metrics"]) == 4
        assert analytics["documents"] >= 20
        assert analytics["stage_metrics"]
        detail = main.stage_alignment_detail(pm_stage, research_stage, model_ids="mpnet-primary")
        assert detail["recommendations"]
        assert 0 <= detail["alignment"] <= 1
        before = db.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        db.execute("DELETE FROM categories WHERE id=?", (cid,))
        db.commit()
        main.seed(db)
        assert db.execute("SELECT COUNT(*) FROM categories").fetchone()[0] == before - 1
        assert db.execute("SELECT COUNT(*) FROM papers WHERE source_path='synthetic demo presence corpus'").fetchone()[0] == 6


def test_chunking_and_vectors_are_deterministic():
    text = "A long research passage about schedule risk and stakeholder evidence. " * 40
    assert len(main.chunks(text)) > 1
    assert main.embed(text, "mpnet-primary") == main.embed(text, "mpnet-primary")
    assert main.embed(text, "mpnet-primary") != main.embed(text, "bge-robustness")
    score = main.cosine(main.embed(text, "mpnet-primary"), main.embed("risk", "mpnet-primary"))
    assert 0 <= score <= 1

