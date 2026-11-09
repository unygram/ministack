"""RDS Data API concurrency regressions; no live AWS validation is claimed.

Busy-transaction errors follow the official Aurora Data API troubleshooting:
https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/data-api.troubleshooting.html
Driver doubles make the scheduling and connection races
deterministic; the data-plane test exercises actual PostgreSQL row locks.
"""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ministack.core.responses import (
    get_account_id,
    get_region,
    set_request_account_id,
    set_request_region,
)
from ministack.services import rds_data

RESOURCE = "arn:aws:rds:us-east-1:000000000000:cluster:concurrency-test"
COMMON = {"resourceArn": RESOURCE, "secretArn": "synthetic-secret", "database": "test"}


class Connection:
    def __init__(self):
        self.execute = Mock()
        self.commit = Mock()
        self.rollback = Mock()
        self.close = Mock()
        self.cursor_close = Mock()
        self.autocommit = True

    def cursor(self):
        return SimpleNamespace(
            execute=self.execute, close=self.cursor_close,
            rowcount=0, lastrowid=None, description=None, statusmessage="SELECT 0",
        )


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(rds_data, "_transactions", {})
    monkeypatch.setattr(rds_data, "_resolve_target", lambda _: (
        {"Endpoint": {"Port": 5432}, "_docker_container_id": "synthetic"},
        "aurora-postgresql", {"HttpEndpointEnabled": True},
    ))
    monkeypatch.setattr(rds_data, "_require_secret_credentials", lambda _: ("user", "password", None))
    connections = []

    def connect(*_args, **_kwargs):
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(rds_data, "_connect", connect)
    yield connections
    rds_data.reset()


async def request(path, **kwargs):
    return await rds_data.handle_request("POST", path, {}, json.dumps({**COMMON, **kwargs}), {})


async def begin():
    status, _, body = await request("/BeginTransaction")
    assert status == 200, body
    return json.loads(body)["transactionId"]


async def health():
    from ministack.app import app

    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    await app({"type": "http", "method": "GET", "path": "/_ministack/health",
               "query_string": b"", "headers": []}, receive, send)
    return sent[0]["status"]


async def wait_started(event):
    async def poll():
        while not event.is_set():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(poll(), 2)


@pytest.mark.parametrize("path", ["/Execute", "/BatchExecute"])
@pytest.mark.parametrize("finish", ["/CommitTransaction", "/RollbackTransaction"])
def test_lock_wait_allows_other_transaction_to_finish(driver, path, finish):
    """A waiting query must not prevent the HTTP request releasing its lock."""
    async def scenario():
        # Exhausting even a one-worker default pool must not trap completion.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        occupied = threading.Event()
        free_pool = threading.Event()
        pool_task = asyncio.create_task(asyncio.to_thread(lambda: (occupied.set(), free_pool.wait(5))))
        await wait_started(occupied)
        holder = await begin()
        waiter = await begin()
        entered, released = threading.Event(), threading.Event()
        driver[0].commit.side_effect = released.set
        driver[0].rollback.side_effect = released.set

        def blocked(*_args):
            entered.set()
            if not released.wait(3):
                raise RuntimeError("lock holder could not finish")

        driver[1].execute.side_effect = blocked
        pending = asyncio.create_task(request(path, transactionId=waiter, sql="SELECT 1"))
        try:
            await wait_started(entered)
            assert await asyncio.wait_for(health(), 1) == 200
            status, _, _ = await asyncio.wait_for(request(finish, transactionId=holder), 2)
            assert status == 200
            status, _, body = await asyncio.wait_for(pending, 2)
            assert status == 200, body
            driver[1].close.assert_not_called()
            assert (await request("/RollbackTransaction", transactionId=waiter))[0] == 200
        finally:
            released.set()
            free_pool.set()
            await pending
            await pool_task
    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/Execute", "/BatchExecute", "/CommitTransaction", "/RollbackTransaction"])
def test_same_transaction_rejects_overlapping_calls(driver, path):
    async def scenario():
        txn = await begin()
        entered, release = threading.Event(), threading.Event()
        driver[0].execute.side_effect = lambda *_: (entered.set(), release.wait(3))
        pending = asyncio.create_task(request("/Execute", transactionId=txn, sql="SELECT 1"))
        try:
            await wait_started(entered)
            status, _, body = await request(path, transactionId=txn, sql="SELECT 2")
            assert status == 400
            assert json.loads(body) == {
                "__type": "DatabaseErrorException", "message": "Transaction is still running a query",
            }
            driver[0].execute.assert_called_once()
            driver[0].commit.assert_not_called()
            driver[0].rollback.assert_not_called()
            driver[0].close.assert_not_called()
        finally:
            release.set()
            await pending
        assert (await request("/CommitTransaction", transactionId=txn))[0] == 200
        driver[0].commit.assert_called_once()
        driver[0].close.assert_called_once()
    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/Execute", "/BatchExecute", "/BeginTransaction"])
def test_slow_connect_does_not_block_transaction_completion(driver, monkeypatch, path):
    async def scenario():
        holder = await begin()
        entered, release = threading.Event(), threading.Event()
        connection = Connection()

        def connect(*_args, **_kwargs):
            entered.set()
            release.wait(3)
            return connection

        monkeypatch.setattr(rds_data, "_connect", connect)
        pending = asyncio.create_task(request(path, sql="SELECT 1"))
        try:
            await wait_started(entered)
            # Also catches holding the transaction-registry mutex during connect.
            assert (await asyncio.wait_for(request("/RollbackTransaction", transactionId=holder), 1))[0] == 200
        finally:
            release.set()
            status, _, _ = await pending
        assert status == 200
    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/CommitTransaction", "/RollbackTransaction"])
def test_completion_keeps_connection_busy_until_driver_returns(driver, path):
    async def scenario():
        txn = await begin()
        entered, release = threading.Event(), threading.Event()
        operation = driver[0].commit if path == "/CommitTransaction" else driver[0].rollback
        operation.side_effect = lambda: (entered.set(), release.wait(3))
        pending = asyncio.create_task(request(path, transactionId=txn))
        try:
            await wait_started(entered)
            status, _, body = await request("/Execute", transactionId=txn, sql="SELECT 1")
            assert status == 400
            assert json.loads(body)["__type"] == "DatabaseErrorException"
            driver[0].close.assert_not_called()
        finally:
            release.set()
            assert (await pending)[0] == 200
        assert txn not in rds_data._transactions
        driver[0].close.assert_called_once()
    asyncio.run(scenario())


def test_cancelled_request_retains_connection_lease_until_sql_finishes(driver):
    async def scenario():
        txn = await begin()
        entered, release = threading.Event(), threading.Event()
        driver[0].execute.side_effect = lambda *_: (entered.set(), release.wait(3))
        pending = asyncio.create_task(request("/Execute", transactionId=txn, sql="SELECT 1"))
        try:
            await wait_started(entered)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            status, _, body = await request("/RollbackTransaction", transactionId=txn)
            assert status == 400
            assert json.loads(body)["__type"] == "DatabaseErrorException"
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await request("/RollbackTransaction", transactionId=txn))[0] == 200
    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/Execute", "/BatchExecute"])
def test_sql_failure_releases_lease_for_rollback(driver, path):
    async def scenario():
        txn = await begin()
        driver[0].execute.side_effect = RuntimeError("synthetic SQL failure")
        assert (await request(path, transactionId=txn, sql="SELECT 1"))[0] == 400
        driver[0].cursor_close.assert_called_once()
        driver[0].close.assert_not_called()
        assert (await request("/RollbackTransaction", transactionId=txn))[0] == 200
        driver[0].close.assert_called_once()
    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/CommitTransaction", "/RollbackTransaction"])
def test_failed_completion_closes_connection(driver, path):
    async def scenario():
        txn = await begin()
        operation = driver[0].commit if path == "/CommitTransaction" else driver[0].rollback
        operation.side_effect = RuntimeError("synthetic completion failure")
        assert (await request(path, transactionId=txn))[0] == 400
        assert txn not in rds_data._transactions
        driver[0].close.assert_called_once()
    asyncio.run(scenario())


def test_reset_defers_busy_connection_close(driver):
    async def scenario():
        txn = await begin()
        entered, release = threading.Event(), threading.Event()
        driver[0].execute.side_effect = lambda *_: (entered.set(), release.wait(3))
        pending = asyncio.create_task(request("/Execute", transactionId=txn, sql="SELECT 1"))
        try:
            await wait_started(entered)
            rds_data.reset()
            assert txn not in rds_data._transactions
            driver[0].close.assert_not_called()
        finally:
            release.set()
            await pending
        driver[0].close.assert_called_once()
    asyncio.run(scenario())


def test_reset_during_connect_does_not_publish_or_leak_transaction(driver, monkeypatch):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        connection = Connection()

        def connect(*_args, **_kwargs):
            entered.set()
            release.wait(3)
            return connection

        monkeypatch.setattr(rds_data, "_connect", connect)
        pending = asyncio.create_task(request("/BeginTransaction"))
        try:
            await wait_started(entered)
            rds_data.reset()
        finally:
            release.set()
            status, _, _ = await pending
        assert status == 504
        assert not rds_data._transactions
        connection.close.assert_called_once()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError, ImportError])
def test_failed_transaction_setup_closes_new_connection(driver, monkeypatch, failure):
    class FailingConnection:
        close = Mock()

        @property
        def autocommit(self):
            return True

        @autocommit.setter
        def autocommit(self, _value):
            raise failure("synthetic driver setup failure")

    connection = FailingConnection()
    monkeypatch.setattr(rds_data, "_connect", lambda *_a, **_k: connection)
    assert asyncio.run(request("/BeginTransaction"))[0] == 400
    connection.close.assert_called_once()
    assert not rds_data._transactions


def test_offloaded_requests_preserve_account_and_region(driver, monkeypatch):
    observed = []
    monkeypatch.setattr(rds_data, "_connect", lambda *_a, **_k: (
        observed.append((get_account_id(), get_region())) or Connection()
    ))

    async def tenant(account, region):
        set_request_account_id(account)
        set_request_region(region)
        txn = await begin()
        assert (await request("/RollbackTransaction", transactionId=txn))[0] == 200

    async def scenario():
        await asyncio.gather(tenant("111111111111", "us-east-1"), tenant("222222222222", "eu-west-1"))

    asyncio.run(scenario())
    assert sorted(observed) == [("111111111111", "us-east-1"), ("222222222222", "eu-west-1")]


@pytest.fixture
def postgres_driver(monkeypatch):
    """Own a disposable PostgreSQL container, independent of any developer DB."""
    docker = pytest.importorskip("docker")
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        client = docker.from_env(timeout=5)
        client.ping()
    except docker.errors.DockerException:
        pytest.skip("Docker is required for the PostgreSQL contention regression")
    container = None
    monitor = None
    try:
        container = client.containers.run(
            "postgres:16-alpine", detach=True,
            environment={"POSTGRES_USER": "test", "POSTGRES_PASSWORD": "synthetic-test-password"},
            ports={"5432/tcp": ("127.0.0.1", None)},
            command=["postgres", "-c", "statement_timeout=5000", "-c", "lock_timeout=4000"],
        )
        container.reload()
        if Path("/.dockerenv").exists():
            host = container.attrs["NetworkSettings"]["Networks"]["bridge"]["IPAddress"]
            port = 5432
        else:
            host = "127.0.0.1"
            port = int(container.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"])
        deadline = time.monotonic() + 30
        while True:
            try:
                monitor = psycopg2.connect(
                    host=host, port=port, user="test", password="synthetic-test-password",
                    dbname="test", connect_timeout=2,
                )
                break
            except psycopg2.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        monitor.autocommit = True
        monkeypatch.setattr(rds_data, "_transactions", {})
        monkeypatch.setattr(rds_data, "_resolve_target", lambda _: (
            {"Endpoint": {"Address": host, "Port": port}, "_docker_container_id": container.id},
            "aurora-postgresql", {"HttpEndpointEnabled": True},
        ))
        monkeypatch.setattr(rds_data, "_require_secret_credentials",
                            lambda _: ("test", "synthetic-test-password", None))
        yield monitor
    finally:
        rds_data.reset()
        if monitor is not None:
            monitor.close()
        if container is not None:
            container.remove(force=True, v=True)
        client.close()


@pytest.mark.data_plane
@pytest.mark.parametrize("finish", ["/CommitTransaction", "/RollbackTransaction"])
def test_postgres_row_lock_releases_through_another_data_api_request(postgres_driver, finish):
    """Actual psycopg2 calls reproduce the check-in drawer's lock dependency."""
    async def scenario():
        assert (await request("/Execute", sql="CREATE TABLE contention (id int PRIMARY KEY, value int)"))[0] == 200
        assert (await request("/Execute", sql="INSERT INTO contention VALUES (1, 0)"))[0] == 200
        holder, waiter = await begin(), await begin()
        assert (await request("/Execute", transactionId=holder,
                              sql="UPDATE contention SET value = value + 1 WHERE id = 1"))[0] == 200
        pending = asyncio.create_task(request("/Execute", transactionId=waiter,
                                              sql="UPDATE contention SET value = value + 1 WHERE id = 1"))

        def waiting_for_lock():
            with postgres_driver.cursor() as cursor:
                cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE wait_event_type = 'Lock')")
                return cursor.fetchone()[0]

        async def wait_for_lock():
            while not await asyncio.to_thread(waiting_for_lock):
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_for_lock(), 2)
            assert await asyncio.wait_for(health(), 1) == 200
            # Unrelated SQL stays responsive while PostgreSQL waits.
            assert (await asyncio.wait_for(request("/Execute", sql="SELECT 1"), 1))[0] == 200
            assert (await asyncio.wait_for(request(finish, transactionId=holder), 1))[0] == 200
            status, _, body = await asyncio.wait_for(pending, 1)
            assert status == 200, body
            assert (await request("/CommitTransaction", transactionId=waiter))[0] == 200
            status, _, body = await request("/Execute", sql="SELECT value FROM contention WHERE id = 1")
            assert status == 200
            assert json.loads(body)["records"] == [[{"longValue": 2 if finish == "/CommitTransaction" else 1}]]
        finally:
            await pending
            for txn in (holder, waiter):
                await request("/RollbackTransaction", transactionId=txn)
    asyncio.run(scenario())
