import json
import multiprocessing

from audit_agentic.rewards.process_judge_client import (
    IMAGE_TAG,
    _append_log,
    render_agent_process,
)


def _append_records(path: str, worker_id: int, count: int) -> None:
    payload = "x" * 16384
    for record_id in range(count):
        _append_log(
            path,
            {
                "worker_id": worker_id,
                "record_id": record_id,
                "payload": payload,
            },
        )


def test_tool_result_images_are_rendered_in_order():
    trace = {
        "system_prompt": "system",
        "context": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "post"},
                    {
                        "type": "data",
                        "source": {"type": "url", "url": "file:///tmp/post.jpg"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "name": "image_tool",
                        "state": "success",
                        "output": [
                            {"type": "text", "text": "before"},
                            {
                                "type": "data",
                                "source": {
                                    "type": "url",
                                    "url": "file:///tmp/tool-a.jpg",
                                },
                            },
                            {"type": "text", "text": "between"},
                            {
                                "type": "data",
                                "source": {
                                    "type": "url",
                                    "url": "file:///tmp/tool-b.jpg",
                                },
                            },
                        ],
                    }
                ],
            },
        ],
    }
    rendered, images, origins = render_agent_process(
        trace=trace,
        prediction=["通过"],
        binary_decision="pass",
        audit_trace="evidence",
        used_rules=[],
        used_tools=["image_tool"],
    )
    assert rendered.count(IMAGE_TAG) == 3
    assert images == [
        "/tmp/post.jpg",
        "/tmp/tool-a.jpg",
        "/tmp/tool-b.jpg",
    ]
    assert origins == ["post", "tool", "tool"]
    image_positions = [
        index
        for index in range(len(rendered))
        if rendered.startswith(IMAGE_TAG, index)
    ]
    assert rendered.index("before") < image_positions[1]
    assert image_positions[1] < rendered.index("between") < image_positions[2]


def test_process_judge_log_is_valid_under_multiprocess_append(tmp_path):
    log_path = tmp_path / "process_judge.jsonl"
    worker_count = 8
    records_per_worker = 24
    workers = [
        multiprocessing.Process(
            target=_append_records,
            args=(str(log_path), worker_id, records_per_worker),
        )
        for worker_id in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == worker_count * records_per_worker
    assert {
        (record["worker_id"], record["record_id"])
        for record in records
    } == {
        (worker_id, record_id)
        for worker_id in range(worker_count)
        for record_id in range(records_per_worker)
    }
