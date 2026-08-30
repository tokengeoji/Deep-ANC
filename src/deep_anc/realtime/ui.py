"""키보드 컨트롤러 — anc_project 패턴 이식 (A/Space 토글, R 리셋, S 저장, Q 종료)."""

from __future__ import annotations

import os
import queue
import select
import sys
import termios
import threading
import tty


class RuntimeState:
    def __init__(self, start_on: bool = False) -> None:
        self.anc_enabled = bool(start_on)
        # 런타임은 무음/ANC OFF로 시작한다. 명시적 사용자 조작 전에는 출력 금지.
        self.noise_enabled = False
        self.quit_event = threading.Event()
        self.reset_event = threading.Event()
        self.save_event = threading.Event()
        self.messages: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self.latest_stats: dict = {}
        self.fatal_error: BaseException | None = None


class KeyboardController(threading.Thread):
    def __init__(self, state: RuntimeState) -> None:
        super().__init__(daemon=True, name="anc-keyboard")
        self.state = state
        self.stop_event = threading.Event()

    @staticmethod
    def help_text() -> str:
        return "A/Space: ANC ON/OFF | N: 소음 ON/OFF | R: 상태 리셋 | Q: 종료 | H: 도움말"

    def handle_key(self, key: str) -> None:
        lowered = key.lower()
        if lowered == "a" or key == " ":
            self.state.anc_enabled = not self.state.anc_enabled
            self.state.messages.put(f"ANC → {'ON' if self.state.anc_enabled else 'OFF'}")
        elif lowered == "n":
            self.state.noise_enabled = not self.state.noise_enabled
            self.state.messages.put(f"소음 스피커 → {'ON' if self.state.noise_enabled else 'OFF'}")
        elif lowered == "r":
            self.state.reset_event.set()
            self.state.messages.put("엔진 상태 리셋 요청")
        elif lowered == "q":
            self.state.quit_event.set()
        elif lowered == "h":
            self.state.messages.put(self.help_text())

    def run(self) -> None:
        if not sys.stdin.isatty():
            while not self.stop_event.is_set() and not self.state.quit_event.is_set():
                line = sys.stdin.readline()
                if not line:
                    return
                for key in line.rstrip("\n"):
                    self.handle_key(key)
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.stop_event.is_set() and not self.state.quit_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.10)
                if readable:
                    data = os.read(fd, 1)
                    if data:
                        self.handle_key(data.decode(errors="ignore"))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def stop(self) -> None:
        self.stop_event.set()
