"""Native Claude transport for MAAS's JSON-sequence Bedrock streaming gateway."""

import base64
import codecs
import copy
import json


MAX_STREAM_BYTES = 32 * 1024 * 1024


class ClaudeStream:
    def __init__(self):
        self.message = None
        self.blocks = {}
        self.inputs = {}
        self.closed = set()
        self.done = False

    def feed(self, envelope):
        if not isinstance(envelope, dict):
            raise ValueError("Claude stream envelope must be an object")
        for name, value in envelope.items():
            if value and (name.endswith("_exception") or name.endswith("Exception")):
                raise RuntimeError("Claude gateway " + name)
        event = envelope
        if "chunk" in envelope:
            chunk = envelope["chunk"]
            if not isinstance(chunk, dict) or not isinstance(chunk.get("bytes"), str):
                raise ValueError("Missing Claude chunk bytes")
            event = json.loads(base64.b64decode(chunk["bytes"], validate=True))
        kind = event.get("type")
        if kind == "error":
            raise RuntimeError("Claude stream error: " + str(event.get("error", {}).get("type", "unknown")))
        if kind == "ping":
            return
        if self.done:
            raise ValueError("Claude events after message_stop")
        if kind == "message":
            if self.message is not None or not event.get("stop_reason"):
                raise ValueError("Invalid complete Claude message")
            self.message = copy.deepcopy(event)
            self.done = True
            return
        if kind == "message_start":
            if self.message is not None:
                raise ValueError("Duplicate Claude message_start")
            self.message = copy.deepcopy(event["message"])
            return
        if self.message is None:
            raise ValueError("Claude stream missing message_start")
        if kind == "content_block_start":
            index = event["index"]
            if index in self.blocks:
                raise ValueError("Duplicate Claude content block")
            self.blocks[index] = copy.deepcopy(event["content_block"])
        elif kind == "content_block_delta":
            index = event["index"]
            if index not in self.blocks or index in self.closed:
                raise ValueError("Claude delta outside an open content block")
            delta = event["delta"]
            block = self.blocks[index]
            key = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}.get(delta["type"])
            if key:
                block[key] = block.get(key, "") + delta[key]
            elif delta["type"] == "input_json_delta":
                self.inputs[index] = self.inputs.get(index, "") + delta["partial_json"]
            elif delta["type"] == "citations_delta":
                block.setdefault("citations", []).append(copy.deepcopy(delta["citation"]))
            else:
                raise ValueError("Unsupported Claude delta type: " + delta["type"])
        elif kind == "content_block_stop":
            index = event["index"]
            if index not in self.blocks or index in self.closed:
                raise ValueError("Invalid Claude content_block_stop")
            if self.inputs.get(index, "").strip():
                value = json.loads(self.inputs[index])
                if not isinstance(value, dict):
                    raise ValueError("Claude tool input must be an object")
                self.blocks[index]["input"] = value
            self.closed.add(index)
        elif kind == "message_delta":
            self.message.update(event["delta"])
            self.message.setdefault("usage", {}).update(event.get("usage") or {})
        elif kind == "message_stop":
            if set(self.blocks) != self.closed or sorted(self.blocks) != list(range(len(self.blocks))):
                raise ValueError("Incomplete Claude content blocks")
            self.message["content"] = [self.blocks[index] for index in sorted(self.blocks)]
            self.done = True
        else:
            raise ValueError("Unsupported Claude event type: " + str(kind))

    def finish(self):
        if not self.done or not self.message or not self.message.get("stop_reason"):
            raise ValueError("Incomplete Claude stream: no completed message")
        return self.message


async def read_claude_response(response):
    # This gateway advertises AWS eventstream but emits concatenated JSON objects.
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    total = 0
    stream = ClaudeStream()

    def consume():
        nonlocal buffer
        buffer = buffer.lstrip()
        while buffer:
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                return
            stream.feed(value)
            buffer = buffer[end:].lstrip()

    async for chunk in response.content.iter_any():
        total += len(chunk)
        if total > MAX_STREAM_BYTES:
            raise ValueError("Claude response exceeds stream size limit")
        buffer += utf8.decode(chunk)
        consume()
    buffer += utf8.decode(b"", final=True)
    consume()
    if buffer.strip():
        raise ValueError("Invalid or truncated MAAS JSON stream")
    return stream.finish()
