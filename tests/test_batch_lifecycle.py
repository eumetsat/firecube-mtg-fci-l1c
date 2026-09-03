# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

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

    ingestor.prepare_batch_data(batch, ctx)
    resources = _resources_for(ingestor, batch.batch_id)
    resources.chunk_owned_cache = MagicMock()
    resources.shared_reader = MagicMock()
    resources.batch_scratch = MagicMock()
    ingestor._batch_registry.register(batch.batch_id, resources.chunk_owned_cache)
    ingestor._batch_registry.register(batch.batch_id, resources.shared_reader)
    ingestor._batch_registry.register(batch.batch_id, resources.batch_scratch)

    ingestor.cleanup_batch_data(batch, ctx)

    assert ingestor._batch_resources == {}
    resources.chunk_owned_cache.close.assert_called_once_with()
    resources.shared_reader.close.assert_called_once_with()
    resources.batch_scratch.close.assert_called_once_with()


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

    ingestor.prepare_batch_data(batch, ctx)
    resources = _resources_for(ingestor, batch.batch_id)
    resources.shared_reader = MagicMock()
    resources.shared_reader.close.side_effect = RuntimeError("decode cleanup failed")
    resources.batch_scratch = MagicMock()
    ingestor._batch_registry.register(batch.batch_id, resources.shared_reader)
    ingestor._batch_registry.register(batch.batch_id, resources.batch_scratch)

    ingestor.cleanup_batch_data(batch, ctx)

    resources.shared_reader.close.assert_called_once_with()
    resources.batch_scratch.close.assert_called_once_with()
    assert ingestor._batch_resources == {}


def test_cleanup_batch_data_runs_after_writer_failure() -> None:
    ingestor = _make_ingestor()
    batch = _make_batch()
    ctx = _make_ctx()

    ingestor.prepare_batch_data(batch, ctx)
    resources = _resources_for(ingestor, batch.batch_id)
    resources.shared_reader = MagicMock()
    resources.batch_scratch = MagicMock()
    ingestor._batch_registry.register(batch.batch_id, resources.shared_reader)
    ingestor._batch_registry.register(batch.batch_id, resources.batch_scratch)

    ingestor.cleanup_batch_data(batch, ctx)

    resources.shared_reader.close.assert_called_once_with()
    resources.batch_scratch.close.assert_called_once_with()
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


def test_orphaned_teardown_continues_after_a_raising_resource() -> None:
    ingestor = _make_ingestor()

    failing = MagicMock()
    failing.close.side_effect = RuntimeError("scratch removal failed")
    healthy = MagicMock()
    ingestor._batch_registry.register("batch-orphan-1", failing)
    ingestor._batch_registry.register("batch-orphan-2", healthy)
    ingestor._batch_resources["batch-orphan-1"] = BatchResources()

    ingestor._teardown_orphaned_batch_resources()

    failing.close.assert_called_once_with()
    healthy.close.assert_called_once_with()
    assert ingestor._batch_resources == {}
