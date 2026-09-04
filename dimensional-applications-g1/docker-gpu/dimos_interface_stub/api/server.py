"""Stub for the missing `dimensional-interface` FastAPI server.

The upstream `dimensional-interface` package referenced by
`dimos.web.robot_web_interface` is not published alongside
`dimos==0.0.12.post2`, which breaks any blueprint pulling in
`WebInput`. This stub satisfies the import chain so the agent stack
(McpServer / McpClient / UnitreeSkillContainer) can still boot;
the browser UI is inert. Send commands with `dimos agent-send "..."`.
"""

from __future__ import annotations

import threading
from typing import Any

import reactivex as rx


class FastAPIServer:
    def __init__(
        self,
        dev_name: str = "Robot Web Interface",
        edge_type: str = "Bidirectional",
        port: int = 5555,
        text_streams: dict[str, Any] | None = None,
        audio_subject: Any = None,
        **streams: Any,
    ) -> None:
        self.dev_name = dev_name
        self.edge_type = edge_type
        self.port = port
        self.text_streams = text_streams or {}
        self.audio_subject = audio_subject
        self.streams = streams
        self.query_stream: rx.subject.Subject[str] = rx.subject.Subject()
        self._stop = threading.Event()

    def run(self) -> None:
        # Blocks until shutdown(); no HTTP server is bound.
        self._stop.wait()

    def shutdown(self) -> None:
        self._stop.set()
