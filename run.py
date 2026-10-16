"""Start the server: `python run.py`. Set FLASK_DEBUG=1 for the reloading dev server."""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    if os.getenv("FLASK_DEBUG", "0") == "1":
        app.run(host=host, port=port, debug=True, threaded=True)
    else:
        from waitress import serve

        print(f"Multi-Agent Console running at http://{host}:{port}")
        # One process only: runs are tracked in memory. Threads carry the SSE streams.
        serve(app, host=host, port=port, threads=16)
