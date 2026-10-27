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
        assert db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 12
        assert db.execute("SELECT COUNT(*) FROM legacy_scores").fetchone()[0] == 576
        assert db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 816
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
        pm_id = db.execute("SELECT id FROM domains WHERE name='Project Management'").fetchone()[0]
        research_stage = db.execute("SELECT id FROM categories WHERE domain_id=? ORDER BY id LIMIT 1", (domain_id,)).fetchone()[0]
        pm_stage = db.execute("SELECT id FROM categories WHERE domain_id=? ORDER BY id LIMIT 1", (pm_id,)).fetchone()[0]
        db.commit()
        bi = main.stage_alignment(pm_id, domain_id, model_ids="mpnet-primary", metric="alignment")
        assert bi["mode"] == "bi"
        assert len(bi["columns"]) == 12
        assert len(bi["rows"]) == 6
        assert len(bi["cells"]) == 72
        detail = main.stage_alignment_detail(pm_stage, research_stage, model_ids="mpnet-primary")
        assert detail["recommendations"]
        assert 0 <= detail["alignment"] <= 1
        before = db.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        db.execute("DELETE FROM categories WHERE id=?", (cid,))
        db.commit()
        main.seed(db)
        assert db.execute("SELECT COUNT(*) FROM categories").fetchone()[0] == before - 1


def test_chunking_and_vectors_are_deterministic():
    text = "A long research passage about schedule risk and stakeholder evidence. " * 40
    assert len(main.chunks(text)) > 1
    assert main.embed(text, "mpnet-primary") == main.embed(text, "mpnet-primary")
    assert main.embed(text, "mpnet-primary") != main.embed(text, "bge-robustness")
    score = main.cosine(main.embed(text, "mpnet-primary"), main.embed("risk", "mpnet-primary"))
    assert 0 <= score <= 1

