import logging

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .config import load_config
from .db import EventHub, Store
from .routes import bp
from .runs import RunManager


def create_app(overrides=None, client_factory=None) -> Flask:
    """Application factory. `client_factory` lets tests inject a fake LLM client."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__)
    app.config.update(load_config())
    app.config.update(overrides or {})

    store = Store(app.config["DATABASE_PATH"])
    hub = EventHub(store)
    app.extensions["store"] = store
    app.extensions["hub"] = hub
    app.extensions["runs"] = RunManager(app.config, store, hub, client_factory)
    app.register_blueprint(bp)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith("/api/"):
            return jsonify({"error": exc.description}), exc.code
        return exc

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    return app
