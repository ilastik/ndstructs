from ndstructs.datasource.precomputed_chunks_info import PrecomputedChunksScale, RawEncoder
from ndstructs.datasource.n5_attributes import GzipCompressor, N5DatasetAttributes, RawCompressor
from ndstructs.datasource.PrecomputedChunksDataSource import PrecomputedChunksDataSource, PrecomputedChunksInfo
from fs.osfs import OSFS
from ndstructs.datasink.precomputed_chunks_sink import PrecomputedChunksScaleSink
from pathlib import Path, PurePosixPath

import pytest
import numpy as np

from ndstructs import Point5D, Array5D, Shape5D
from ndstructs.datasource.DataSource import DataRoi, ArrayDataSource, DataSource
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
        outer_path=Path("test_n5_datasink.n5"),
        inner_path=PurePosixPath("/data"),
        attributes=N5DatasetAttributes(
            dimensions=datasource.shape,
            blockSize=Shape5D(x=10, y=10),
            axiskeys=datasource.axiskeys,
            dataType=datasource.dtype,
            compression=RawCompressor(),
            location=Point5D.zero(x=7, y=13)
        )
    )
    for tile in DataRoi(datasource).split(sink.tile_shape):
        sink.write(tile.retrieve().translated(Point5D.zero(x=7, y=13)))

    n5ds = N5DataSource(filesystem=sink.filesystem, path=sink.path)
    saved_data = n5ds.retrieve()
    assert saved_data.location == Point5D.zero(x=7, y=13)
    assert saved_data == data

def test_distributed_n5_datasink(tmp_path: Path, data: Array5D, datasource: DataSource):
    filesystem = OSFS(tmp_path.as_posix())
    outer_path = Path("test_distributed_n5_datasink.n5")
    inner_path = PurePosixPath("/data")
    full_path = Path("test_distributed_n5_datasink.n5/data")
    attributes = N5DatasetAttributes(
        dimensions=datasource.shape,
        blockSize=datasource.tile_shape,
        axiskeys=datasource.axiskeys,
        dataType=datasource.dtype,
        compression=RawCompressor()
    )
    sinks = [
        N5DatasetSink.create(outer_path=outer_path, inner_path=inner_path, filesystem=filesystem, attributes=attributes),
        N5DatasetSink.open(path=full_path, filesystem=filesystem),
        N5DatasetSink.open(path=full_path, filesystem=filesystem),
        N5DatasetSink.open(path=full_path, filesystem=filesystem),
    ]

    for idx, piece in enumerate(DataRoi(datasource).default_split()):
        sink = sinks[idx % len(sinks)]
        sink.write(piece.retrieve())

    n5ds = N5DataSource(filesystem=filesystem, path=full_path)
    assert n5ds.retrieve() == data

def test_writing_to_precomputed_chunks(tmp_path: Path, data: Array5D):
    datasource = ArrayDataSource.from_array5d(data, tile_shape=Shape5D(x=10, y=10))
    scale = PrecomputedChunksScale.from_datasource(datasource=datasource, key=Path("my_test_data"), encoding=RawEncoder())
    info = PrecomputedChunksInfo(
        data_type=datasource.dtype,
        type_="image",
        num_channels=datasource.shape.c,
        scales=tuple([scale]),
    )
    sink_path = Path("mytest.precomputed")
    filesystem = OSFS(tmp_path.as_posix())
    datasink = PrecomputedChunksScaleSink.create(
        path=sink_path,
        filesystem=filesystem,
        info=info,
        resolution=scale.resolution,
    )

    for tile in datasource.roi.get_datasource_tiles():
        datasink.write(tile.retrieve())

    precomp_datasource = PrecomputedChunksDataSource(path=sink_path, filesystem=filesystem, resolution=scale.resolution)
    reloaded_data = precomp_datasource.retrieve()
    assert reloaded_data == data


def test_writing_to_offset_precomputed_chunks(tmp_path: Path, data: Array5D):
    datasource = ArrayDataSource.from_array5d(data, tile_shape=Shape5D(x=10, y=10), location=Point5D(x=1000, y=1000))
    scale = PrecomputedChunksScale.from_datasource(datasource=datasource, key=Path("my_test_data"), encoding=RawEncoder())
    sink_path = Path("mytest.precomputed")
    filesystem = OSFS(tmp_path.as_posix())
    datasink = PrecomputedChunksScaleSink.create(
        path=sink_path,
        filesystem=filesystem,
        resolution=scale.resolution,
        info=PrecomputedChunksInfo(
            data_type=datasource.dtype,
            type_="image",
            num_channels=datasource.shape.c,
            scales=tuple([scale]),
        ),
    )

    for tile in datasource.roi.get_datasource_tiles():
        datasink.write(tile.retrieve())

    precomp_datasource = PrecomputedChunksDataSource(path=sink_path, filesystem=filesystem, resolution=scale.resolution)
    reloaded_data = precomp_datasource.retrieve()
    assert (reloaded_data.raw("xyz") == data.raw("xyz")).all() # type: ignore
