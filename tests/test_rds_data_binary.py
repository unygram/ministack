"""Binary response regressions, based on the AWS Field shape and local PostgreSQL.

https://docs.aws.amazon.com/rdsdataservice/latest/APIReference/API_Field.html
These tests do not claim validation against a real AWS account.
"""

import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ministack.services import rds_data


def configure_database(monkeypatch, host="unused", port=5432):
    monkeypatch.setattr(rds_data, "_resolve_target", lambda _: (
        {"Endpoint": {"Address": host, "Port": port}, "_docker_container_id": "binary-test"},
        "aurora-postgresql", {"HttpEndpointEnabled": True},
    ))
    monkeypatch.setattr(rds_data, "_require_secret_credentials", lambda _: (
        "test", "synthetic-test-password", None,
    ))


def execute(sql):
    return rds_data._execute_statement({
        "resourceArn": "arn:aws:rds:us-east-1:000000000000:cluster:binary-test",
        "secretArn": "synthetic-secret", "database": "test", "sql": sql,
    })


@pytest.mark.parametrize("binary_type", [bytes, bytearray, memoryview])
@pytest.mark.parametrize("payload", [b"", b"\x00\x01\xfe\xff", bytes(range(256))])
def test_execute_returns_binary_fields_without_stringifying(monkeypatch, binary_type, payload):
    configure_database(monkeypatch)
    cursor = SimpleNamespace(
        execute=Mock(), close=Mock(), rowcount=1, lastrowid=None, statusmessage="SELECT 1",
        description=[("value",)],
        fetchall=lambda: [(binary_type(payload), None, "plain text", 42, True)],
    )
    connection = SimpleNamespace(cursor=lambda: cursor, close=Mock())
    monkeypatch.setattr(rds_data, "_connect", lambda *args, **kwargs: connection)

    status, _, body = execute("SELECT binary_value, nullable_value, text_value, int_value, bool_value")

    assert status == 200
    assert json.loads(body)["records"] == [[
        {"blobValue": base64.b64encode(payload).decode()},
        {"isNull": True}, {"stringValue": "plain text"}, {"longValue": 42}, {"booleanValue": True},
    ]]
    cursor.close.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.data_plane
def test_postgres_bytea_returns_blob_value(monkeypatch):
    """Exercise psycopg2's actual memoryview result without touching developer data."""
    docker = pytest.importorskip("docker")
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        client = docker.from_env(timeout=5)
        client.ping()
    except docker.errors.DockerException:
        pytest.skip("Docker is required for the PostgreSQL binary regression")
    container = None
    try:
        container = client.containers.run(
            "postgres:16-alpine", detach=True,
            environment={"POSTGRES_USER": "test", "POSTGRES_PASSWORD": "synthetic-test-password"},
            ports={"5432/tcp": ("127.0.0.1", None)},
        )
        container.reload()
        if Path("/.dockerenv").exists():
            host, port = container.attrs["NetworkSettings"]["Networks"]["bridge"]["IPAddress"], 5432
        else:
            host = "127.0.0.1"
            port = int(container.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"])
        deadline = time.monotonic() + 30
        while True:
            try:
                connection = psycopg2.connect(
                    host=host, port=port, user="test", password="synthetic-test-password",
                    dbname="test", connect_timeout=2,
                )
                connection.close()
                break
            except psycopg2.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        configure_database(monkeypatch, host, port)

        status, _, body = execute("SELECT decode('0001feff', 'hex'), decode('', 'hex'), NULL::bytea")

        assert status == 200, body
        assert json.loads(body)["records"] == [[
            {"blobValue": "AAH+/w=="}, {"blobValue": ""}, {"isNull": True},
        ]]
    finally:
        if container is not None:
            container.remove(force=True, v=True)
        client.close()
