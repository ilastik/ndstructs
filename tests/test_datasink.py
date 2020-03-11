from pathlib import Path

import pytest
import numpy as np

from ndstructs import Point5D, Slice5D, Array5D, Shape5D
from ndstructs.datasource.DataSource import ArrayDataSource, DataSource
from ndstructs.datasource.DataSourceSlice import DataSourceSlice
from ndstructs.datasource.N5DataSource import N5DataSource
from ndstructs.datasink import N5DataSink


@pytest.fixture
def data() -> Array5D:
    return Array5D(np.arange(20 * 10 * 7).reshape(20, 10, 7), axiskeys="xyz")


@pytest.fixture
def datasource(data: Array5D):
    return ArrayDataSource.from_array5d(data, tile_shape=Shape5D(x=10, y=10))


def test_n5_datasink(tmp_path: Path, data: Array5D, datasource: DataSource):
    dataset_path = tmp_path / "test_n5_datasink.n5/data"
    sink = N5DataSink(path=dataset_path, data_slice=DataSourceSlice(datasource), tile_shape=Shape5D(x=10, y=10))
    sink.process(Slice5D.all())

    n5ds = DataSource.create(dataset_path)
    assert n5ds.retrieve(Slice5D.all()) == data


def test_n5_datasink_saves_roi(tmp_path: Path, data: Array5D, datasource: DataSource):
    roi = DataSourceSlice(datasource, x=slice(5, 8), y=slice(2, 4))

    dataset_path = tmp_path / "test_n5_datasink_saves_roi.n5/data"
    sink = N5DataSink(path=dataset_path, data_slice=roi, tile_shape=Shape5D(x=10, y=10))
    sink.process(Slice5D.all())

    n5ds = DataSource.create(dataset_path)
    assert n5ds.retrieve(Slice5D.all()) == roi.retrieve()


def test_distributed_n5_datasink(tmp_path: Path, data: Array5D, datasource: DataSource):
    dataset_path = tmp_path / "test_distributed_n5_datasink.n5/data"
    data_slice = DataSourceSlice(datasource)

    sinks = [
        N5DataSink(path=dataset_path, data_slice=data_slice, mode=N5DataSink.Mode.CREATE),
        N5DataSink(path=dataset_path, data_slice=data_slice, mode=N5DataSink.Mode.OPEN),
        N5DataSink(path=dataset_path, data_slice=data_slice, mode=N5DataSink.Mode.OPEN),
        N5DataSink(path=dataset_path, data_slice=data_slice, mode=N5DataSink.Mode.OPEN),
    ]

    for idx, piece in enumerate(data_slice.split()):
        sink = sinks[idx % len(sinks)]
        sink.process(piece)

    n5ds = DataSource.create(dataset_path)
    assert n5ds.retrieve(Slice5D.all()) == data
