from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from firecube_mtg_fci_l1c.config import MtgFciL1cConfig
from firecube_mtg_fci_l1c.ingestor import BatchResources, MtgFciL1cIngestor


def _make_batch(batch_id: str = "batch-001") -> Any:
    return SimpleNamespace(batch_id=batch_id)


def _make_ctx() -> Any:
    return SimpleNamespace()


def _make_ingestor() -> MtgFciL1cIngestor:
    ingestor = MtgFciL1cIngestor()
    ingestor.plugin_config = MtgFciL1cConfig(product_type="FDHSI", time_slots=144)
    return ingestor


def _resources_for(ingestor: MtgFciL1cIngestor, batch_id: str) -> BatchResources:
    return cast(BatchResources, ingestor._batch_resources[batch_id])


def test_cleanup_batch_data_removes_registered_resources_on_success() -> None:
    ingestor = _make_ingestor()
    batch = _make_batch()
    ctx = _make_ctx()
    cleanup_called = threading.Event()

    ingestor.prepare_batch_data(batch, ctx)
    resources = _resources_for(ingestor, batch.batch_id)
    resources.shared_reader = MagicMock()
    resources.batch_scratch = MagicMock(side_effect=None)
    resources.batch_scratch.cleanup.side_effect = cleanup_called.set
    ingestor._retained_batch_scratches.extend(
        [resources.shared_reader, resources.batch_scratch]
    )

    ingestor.cleanup_batch_data(batch, ctx)

    assert cleanup_called.wait(1)
    assert ingestor._batch_resources == {}
    assert ingestor._retained_batch_scratches == []
    resources.shared_reader.close.assert_called_once_with()
    resources.batch_scratch.cleanup.assert_called_once_with()


def test_cleanup_batch_data_is_idempotent() -> None:
    ingestor = _make_ingestor()
    batch = _make_batch()
    ctx = _make_ctx()

    ingestor.prepare_batch_data(batch, ctx)
    ingestor.cleanup_batch_data(batch, ctx)
    ingestor.cleanup_batch_data(batch, ctx)

    assert ingestor._batch_resources == {}


def test_cleanup_batch_data_runs_after_decode_failure() -> None:
    ingestor = _make_ingestor()
    batch = _make_batch()
    ctx = _make_ctx()
    cleanup_called = threading.Event()

    ingestor.prepare_batch_data(batch, ctx)
    resources = _resources_for(ingestor, batch.batch_id)
    resources.shared_reader = MagicMock()
    resources.shared_reader.close.side_effect = RuntimeError("decode cleanup failed")
    resources.batch_scratch = MagicMock()
    resources.batch_scratch.cleanup.side_effect = cleanup_called.set

    ingestor.cleanup_batch_data(batch, ctx)

    resources.shared_reader.close.assert_called_once_with()
    assert cleanup_called.wait(1)
    resources.batch_scratch.cleanup.assert_called_once_with()
    assert ingestor._batch_resources == {}


def test_cleanup_batch_data_runs_after_writer_failure() -> None:
    ingestor = _make_ingestor()
    batch = _make_batch()
    ctx = _make_ctx()
    cleanup_called = threading.Event()

    ingestor.prepare_batch_data(batch, ctx)
    resources = _resources_for(ingestor, batch.batch_id)
    resources.shared_reader = MagicMock()
    resources.batch_scratch = MagicMock()
    resources.batch_scratch.cleanup.side_effect = cleanup_called.set

    ingestor.cleanup_batch_data(batch, ctx)

    resources.shared_reader.close.assert_called_once_with()
    assert cleanup_called.wait(1)
    resources.batch_scratch.cleanup.assert_called_once_with()
    assert ingestor._batch_resources == {}


def test_concurrent_batches_cannot_remove_each_other_resources() -> None:
    ingestor = _make_ingestor()
    batch_a = _make_batch("batch-a")
    batch_b = _make_batch("batch-b")
    ctx = _make_ctx()

    ingestor.prepare_batch_data(batch_a, ctx)
    ingestor.prepare_batch_data(batch_b, ctx)

    ingestor.cleanup_batch_data(batch_a, ctx)

    assert "batch-a" not in ingestor._batch_resources
    assert "batch-b" in ingestor._batch_resources


def test_second_pipeline_invocation_starts_without_stale_readers() -> None:
    ingestor = _make_ingestor()
    first_batch = _make_batch("first-batch")
    second_batch = _make_batch("second-batch")
    ctx = _make_ctx()

    ingestor.prepare_batch_data(first_batch, ctx)
    ingestor.cleanup_batch_data(first_batch, ctx)
    ingestor.prepare_batch_data(second_batch, ctx)

    assert set(ingestor._batch_resources) == {"second-batch"}
