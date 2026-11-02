"""One-command local run.

    python -m app --demo            # offline sample collections, http://localhost:8000
    python -m app                   # your documents (downloads the multilingual e5 model once)
    python -m app --demo --port 18000 --host 0.0.0.0
"""
import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m app", description="Run the Vouch server.")
    p.add_argument("--demo", action="store_true", help="offline demo: no API keys, no model download")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = p.parse_args()
    if args.demo:
        os.environ["VOUCH_DEMO"] = "1"
    import uvicorn

    print(f"Vouch on http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
          f"  (desk: /app, console: /admin){'  [demo]' if args.demo else ''}")
    uvicorn.run("app.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
