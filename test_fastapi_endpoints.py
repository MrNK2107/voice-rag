import os
import sys
from fastapi.testclient import TestClient

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

from app.main import app


def test_api_endpoints():
    print("==================================================")
    print("  Testing FastAPI App Endpoints")
    print("==================================================")

    with TestClient(app) as client:
        print("\n1. Testing GET /health ...")
        res = client.get("/health")
        print(f"Health Status Code: {res.status_code}")
        print(f"Health Response: {res.json()}")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"
        assert res.json()["harness_loaded"] is True

        print("\n2. Testing GET / (Static Web UI route) ...")
        res = client.get("/")
        print(f"Root UI Status Code: {res.status_code}")
        assert res.status_code == 200

        print("\n3. Testing POST /api/ask-text ...")
        res = client.post("/api/ask-text", json={"query": "मैकडॉनल्ड्स क्या है?"})
        print(f"Ask-Text Status Code: {res.status_code}")
        data = res.json()
        print(f"Transcript: {data.get('transcript')}")
        print(f"Answer: {data.get('answer')}")
        print(f"Abstained: {data.get('abstained')} (Reason: {data.get('abstain_reason')})")
        print(f"Timings ms: {data.get('timings_ms')}")
        assert res.status_code == 200

        print("\n4. Testing POST /api/ask-text with Off-Topic Query ...")
        res = client.post("/api/ask-text", json={"query": "What is the distance to Mars?"})
        data = res.json()
        print(f"Answer: {data.get('answer')}")
        print(f"Abstained: {data.get('abstained')} (Reason: {data.get('abstain_reason')})")
        assert data.get('abstained') is True

        print("\n5. Testing POST /api/ask-audio ...")
        dummy_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00\x77\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        files = {"file": ("test.wav", dummy_audio, "audio/wav")}
        res = client.post("/api/ask-audio", files=files)
        print(f"Ask-Audio Status Code: {res.status_code}")
        data = res.json()
        print(f"Transcript: {data.get('transcript')}")
        print(f"Answer: {data.get('answer')}")
        print(f"Timings ms: {data.get('timings_ms')}")
        assert res.status_code == 200

        print("\n[OK] All FastAPI Endpoint tests completed successfully!")


if __name__ == "__main__":
    test_api_endpoints()

