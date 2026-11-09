"""reflex-bot must act on SIGTERM (reflex-6b36).

Without a handler this process -- PID 1 in its container -- has the signal
dropped by the kernel, so only the SIGKILL at the end of the 60s stop grace
could stop it. That grace is longer than the docker daemon's own 15s shutdown
budget, so a daemon restart left the container alive, killed it on the next
daemon start, and never engaged the restart policy. The container measurements
are on reflex-6b36.

What these tests do NOT reproduce is PID 1 itself: pytest is not init, so the
default disposition applies here and would kill a child on its own. That is
why the pair below measures BOTH branches -- exit 0 with the handler and
-SIGTERM without -- because exit 0 on its own would prove nothing.
"""

from __future__ import annotations

import select
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# Only the bot and the webhook server are stubbed, so the REAL run_bot runs:
# its `async with bot` and `await bot.start(...)` are the code under test, and
# __aexit__ printing "closed" is how the test knows the bot was closed rather
# than the process merely dying.
_CHILD = """
import asyncio
import reflex.services.bot as bot

class _Bot:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        print("closed", flush=True)
    async def start(self, token):
        await asyncio.sleep(30)

async def _serve(b, port):
    print("ready", flush=True)

bot.start_metrics_server = lambda port: None
bot.ReflexBot = _Bot
bot.start_webhook_server = _serve
INSTALL
bot.main()
"""


def _child(tmp_path: Path, install: bool) -> tuple[int, str]:
    """Run main() in a subprocess, SIGTERM it, return (exit code, stdout)."""
    script = tmp_path / "child.py"
    script.write_text(
        _CHILD.replace("INSTALL", "" if install else "bot.signal.signal = lambda *a: None")
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        text=True,
        env={"DISCORD_BOT_TOKEN": "x", "PATH": "/usr/bin:/bin"},
    )
    try:
        assert proc.stdout is not None
        # Bounded: readline() on a child that hangs before printing would
        # otherwise block until the CI job's own limit, with no diagnostic.
        assert select.select([proc.stdout], [], [], 30)[0], "child never spoke"
        assert proc.stdout.readline().strip() == "ready"
        proc.terminate()
        code = proc.wait(timeout=30)
        return code, proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


@pytest.fixture
def restore_sigterm() -> Iterator[None]:
    """Keep a test's handler out of the rest of the session."""
    previous = signal.getsignal(signal.SIGTERM)
    yield
    # getsignal returns None for a handler installed from C, and signal.signal
    # refuses None.
    signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


def test_sigterm_exits_the_process_cleanly(tmp_path: Path) -> None:
    code, out = _child(tmp_path, install=True)
    assert code == 0
    # Not just "the process ended": the bot's context manager exited, which is
    # what closes the Discord session.
    assert "closed" in out


def test_without_the_handler_sigterm_kills_the_process(tmp_path: Path) -> None:
    # The control. The two outcomes have to differ, or exit 0 above would say
    # nothing about the handler.
    code, out = _child(tmp_path, install=False)
    assert code == -signal.SIGTERM
    assert "closed" not in out


def test_handler_raises_and_disarms_itself(restore_sigterm: None) -> None:
    import reflex.services.bot as bot

    # The raise is what main() exits on. The SIG_IGN is so a second SIGTERM
    # cannot re-raise inside asyncio's cleanup and abort it.
    with pytest.raises(KeyboardInterrupt):
        bot._shutdown_on_sigterm(15, None)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
