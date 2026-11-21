"""Shared test settings.

Database selection in tests: api.db.get_database_url() returns TEST_DATABASE_URL first,
then DATABASE_URL, then in-memory SQLite. api/db.py also calls load_dotenv(), which
would otherwise pull the MySQL DATABASE_URL from a developer's .env, so TEST_DATABASE_URL
is forced to in-memory SQLite here, before anything imports api.db. A session-start check
refuses to run if the engine is not SQLite.
API key: an explicit throwaway value; rate limiter off unless a test turns it on.
"""

import os

os.environ["TEST_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["API_KEY"] = "test-api-key"
os.environ["RATE_LIMIT"] = "0"


def _engine_url() -> str:
    from api.db import engine

    return str(engine.url)


def pytest_report_header(config):
    return f"database: TEST_DATABASE_URL={os.environ['TEST_DATABASE_URL']} -> engine {_engine_url()}"


def pytest_sessionstart(session):
    from api.db import engine

    if engine.dialect.name != "sqlite":
        raise RuntimeError(f"tests must use SQLite, engine is {engine.url}")
