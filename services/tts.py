import asyncio
import os
from io import BytesIO
from typing import Optional

import edge_tts

EDGE_VOICE = os.getenv("SUNSUNA_EDGE_VOICE", "ne-NP-HemkalaNeural")


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    return asyncio.run(coro)


def synthesize_edge_tts(text: str, voice: Optional[str] = None) -> bytes:
    async def _synth() -> bytes:
        buffer = BytesIO()
        communicate = edge_tts.Communicate(text=text, voice=voice or EDGE_VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()

    return _run_async(_synth())
