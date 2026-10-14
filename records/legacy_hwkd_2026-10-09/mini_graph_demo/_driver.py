"""Shared Bolt driver factory.

Centralised so seed/queries/export/verifiers all read the same env var and
endpoint. If the env var is missing we fail fast with a one-line message.
"""
from __future__ import annotations

import os
import sys

from neo4j import GraphDatabase, Driver

URI = "bolt://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD_ENV = "LOCALML_SCHEDULER_NEO4J_PASSWORD"
DATABASE = "neo4j"


def connect() -> Driver:
    password = os.environ.get(PASSWORD_ENV)
    if not password:
        sys.stderr.write(
            f"error: env var {PASSWORD_ENV} not set; "
            f"`export {PASSWORD_ENV}=<password>` and retry\n"
        )
        sys.exit(2)
    driver = GraphDatabase.driver(URI, auth=(USERNAME, password))
    driver.verify_connectivity()
    return driver


def session(driver: Driver):
    return driver.session(database=DATABASE)
