from os import path

from attr import attr
from ndstructs.datasource.n5_attributes import GzipCompressor, N5DatasetAttributes, RawCompressor
from ndstructs.datasource.PrecomputedChunksDataSource import PrecomputedChunksDataSource, PrecomputedChunksInfo
from fs.osfs import OSFS
from ndstructs.datasink.PrecomputedChunksDataSink import PrecomputedChunksDataSink
from pathlib import Path

import pytest
import numpy as np

from ndstructs import Point5D, Interval5D, Array5D, Shape5D
from ndstructs.datasource.DataSource import ArrayDataSource, DataSource
from ndstructs.datasource.DataRoi import DataRoi
from ndstructs.datasource.N5DataSource import N5DataSource
from ndstructs.datasink.n5_dataset_sink import N5DatasetSink


@pytest.fixture
def data() -> Array5D:
    array = Array5D(np.arange(20 * 10 * 7).reshape(20, 10, 7), axiskeys="xyz")
    array.setflags(write=False)
    return array

@pytest.fixture
def datasource(data: Array5D) -> DataSource:
    return ArrayDataSource.from_array5d(data, tile_shape=Shape5D(x=10, y=10))

def test_n5_attributes():
    attributes = N5DatasetAttributes(
        dimensions=Shape5D(x=100, y=200),
        blockSize=Shape5D(x=10, y=20),
        axiskeys="yx",
        dataType=np.dtype("uint16").newbyteorder(">"), #type: ignore
        compression=GzipCompressor(level=3)
    )

    reserialized_attributes = N5DatasetAttributes.from_json_data(attributes.to_json_data())
    assert reserialized_attributes == attributes
    assert attributes.to_json_data()["axes"] == ("x", "y")

def test_n5_datasink(tmp_path: Path, data: Array5D, datasource: DataSource):
    sink = N5DatasetSink.create(
        filesystem=OSFS(tmp_path.as_posix()),
        path=Path("test_n5_datasink.n5/data"),
        attributes=N5DatasetAttributes(
            dimensions=datasource.shape,
            blockSize=Shape5D(x=10, y=10),
            axiskeys=datasource.axiskeys,
            dataType=datasource.dtype,
            compression=RawCompressor()
        )
    )
    for tile in DataRoi(datasource).split(sink.tile_shape):
        sink.write(tile.retrieve())

    n5ds = DataSource.create(filesystem=sink.filesystem, path=sink.path)
    assert n5ds.retrieve() == data

def test_distributed_n5_datasink(tmp_path: Path, data: Array5D, datasource: DataSource):
    filesystem = OSFS(tmp_path.as_posix())
    path = Path("test_distributed_n5_datasink.n5/data")
    attributes = N5DatasetAttributes(
        dimensions=datasource.shape,
        blockSize=datasource.tile_shape,
        axiskeys=datasource.axiskeys,
        dataType=datasource.dtype,
        compression=RawCompressor()
    )
    sinks = [
        N5DatasetSink.create(path=path, filesystem=filesystem, attributes=attributes),
        N5DatasetSink.open(path=path, filesystem=filesystem),
        N5DatasetSink.open(path=path, filesystem=filesystem),
        N5DatasetSink.open(path=path, filesystem=filesystem),
    ]

    for idx, piece in enumerate(DataRoi(datasource).split()):
        sink = sinks[idx % len(sinks)]
        sink.write(piece.retrieve())

    n5ds = DataSource.create(filesystem=filesystem, path=path)
    assert n5ds.retrieve() == data

def test_writing_to_precomputed_chunks(tmp_path: Path, data: Array5D):
    tile_shape = tile_shape=Shape5D(x=10, y=10)
    datasource = ArrayDataSource.from_array5d(data, tile_shape=tile_shape)
    info = PrecomputedChunksInfo.from_datasource(
        scale_key="my_test_data",
        datasource=datasource,
        resolution=(1,1,1)
    )
    filesystem = OSFS(tmp_path.as_posix())
    root_path = Path("mytest.precomputed")
    datasink = PrecomputedChunksDataSink.create(
        root_path=root_path, filesystem=filesystem, info=info
    )

    for tile in DataRoi(datasource).get_tiles():
        datasink.write(scale=info.scales[0], chunk=tile.retrieve())

    precomp_datasource = PrecomputedChunksDataSource.create(path=root_path / info.scales[0].key, filesystem=filesystem)
    reloaded_data = DataRoi(precomp_datasource).retrieve()
    assert reloaded_data == data
