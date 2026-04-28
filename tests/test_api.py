def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_kb_and_chat(client):
    r = client.post("/v1/knowledge_bases", json={"name": "demo", "description": "试"})
    assert r.status_code == 200
    kb = r.json()["id"]
    sample = "\u8fd9\u662f\u4e00\u6bb5\u6d4b\u8bd5\u3002\u8425\u6536\u540c\u6bd4\u589e\u957f10%\u3002"
    up = client.post(
        f"/v1/knowledge_bases/{kb}/documents",
        files={"file": ("notes.txt", sample.encode("utf-8"), "text/plain")},
    )
    assert up.status_code == 202, up.text
    import time

    job_id = up.json()["job_id"]
    j = {"status": "unknown"}
    for _ in range(100):
        j = client.get(f"/v1/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    r = client.post(
        "/v1/chat",
        json={
            "knowledge_base_id": kb,
            "message": "\u5185\u5bb9\u662f\u4ec0\u4e48",
            "history": [],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "answer" in body
    assert body["citations"]


def test_404(client):
    r = client.get("/v1/knowledge_bases/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
