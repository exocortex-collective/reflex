"""reflex-bot must act on SIGTERM (reflex-6b36).

Without a handler this process -- PID 1 in its container -- has the signal
dropped by the kernel, so nothing but SIGKILL could stop it. The container
measurements are on reflex-6b36.

pytest is not init, so none of this reproduces that drop: here the default
disposition would kill a child unaided. That is why the pair below measures
BOTH branches -- exit 0 with the handler, -SIGTERM without -- since exit 0 on
its own would say nothing about the handler.
"""

from __future__ import annotations

import select
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

# Only the bot and the webhook server are stubbed, so the REAL run_bot runs:
# its `async with bot` and `await bot.start(...)` are the code under test, and
# __aexit__ printing "closed" is how a closed bot is told apart from a process
# that merely died.
_CHILD = """
import asyncio
import reflex.services.bot as bot

class _Bot:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        await asyncio.sleep(AEXIT_SLEEP)
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


def _child(
    tmp_path: Path,
    install: bool = True,
    aexit_sleep: float = 0.0,
    signals: int = 1,
) -> tuple[int, str]:
    """Run main() in a subprocess, SIGTERM it, return (exit code, stdout).

    `signals` > 1 sends the extra SIGTERMs while __aexit__ is still running,
    which is what `aexit_sleep` buys time for.
    """
    script = tmp_path / "child.py"
    script.write_text(
        _CHILD.replace("AEXIT_SLEEP", repr(aexit_sleep)).replace(
            "INSTALL", "" if install else "bot.signal.signal = lambda *a: None"
        )
    )
    with subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        text=True,
        env={"DISCORD_BOT_TOKEN": "x", "PATH": "/usr/bin:/bin"},
    ) as proc:
        try:
            assert proc.stdout is not None
            # Bounded: readline() on a child that hangs before printing would
            # otherwise block forever -- there is no pytest-timeout here.
            assert select.select([proc.stdout], [], [], 30)[0], "child never spoke"
            assert proc.stdout.readline().strip() == "ready"
            for _ in range(signals):
                proc.terminate()
                time.sleep(0.2)
            code = proc.wait(timeout=30)
            return code, proc.stdout.read()
        finally:
            if proc.poll() is None:
                proc.kill()


@pytest.fixture
def restore_sigterm() -> Iterator[None]:
    """Keep a test's handler out of the rest of the session."""
    previous = signal.getsignal(signal.SIGTERM)
    yield
    # getsignal returns None for a handler installed from C, and signal.signal
    # refuses None.
    signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


def test_sigterm_exits_the_process_cleanly(tmp_path: Path) -> None:
    code, out = _child(tmp_path)
    assert code == 0
    assert "closed" in out


def test_without_the_handler_sigterm_kills_the_process(tmp_path: Path) -> None:
    code, out = _child(tmp_path, install=False)
    assert code == -signal.SIGTERM
    assert "closed" not in out


def test_a_repeat_sigterm_does_not_abort_the_shutdown(tmp_path: Path) -> None:
    # The behaviour the handler's SIG_IGN exists for, rather than the fact that
    # it was called: three SIGTERMs land while __aexit__ is mid-flight, and the
    # close still completes. Without the disarm the second one raises inside
    # asyncio's cleanup and "closed" is never printed.
    code, out = _child(tmp_path, aexit_sleep=1.0, signals=3)
    assert code == 0
    assert "closed" in out


def test_handler_raises_and_disarms_itself(restore_sigterm: None) -> None:
    import reflex.services.bot as bot

    with pytest.raises(KeyboardInterrupt):
        bot._shutdown_on_sigterm(15, None)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
