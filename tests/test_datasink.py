from pathlib import Path

import pytest
import numpy as np

from ndstructs import Point5D, Slice5D, Array5D
from ndstructs.datasource.DataSource import ArrayDataSource
from ndstructs.datasource.N5DataSource import N5DataSource
from ndstructs.datasink import N5DataSink


def test_n5_datasink(tmp_path: Path):
    data = Array5D(np.arange(20 * 10).reshape(20, 10), axiskeys="xy")
    ds = ArrayDataSource(data=data)
    dataset_path = tmp_path / "test_n5_datasink.n5/data"
    sink = N5DataSink(url=dataset_path, datasource=ds)
    sink.process(Slice5D.all())

    n5ds = N5DataSource(dataset_path)
    assert n5ds.retrieve(Slice5D.all()) == data
