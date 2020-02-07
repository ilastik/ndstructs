import pytest
from pathlib import Path
import os
import numpy as np
import z5py

from ndstructs.datasource import DataSourceUrl


def test_is_archive_path():
    assert DataSourceUrl.is_archive_path(Path("blas/bles/blus.h5/some_dataset"))
    assert not DataSourceUrl.is_archive_path(Path("blas/bles/blus.h5"))
    assert DataSourceUrl.is_archive_path(Path("/blas/*/bles/blus.h5/some/internal/path"))

    assert DataSourceUrl.is_archive_path(Path("blas/bles/blus.n5/some_Dataset"))
    assert not DataSourceUrl.is_archive_path(Path("blas/bles/blus.n5"))
    assert DataSourceUrl.is_archive_path(Path("/blas/*/bles/blus.N5/some/internal/path"))


def test_globbing(tmp_path):
    dirs = [Path("a"), Path("a/b"), Path("a/b/c"), Path("a/b/c/d"), Path("a/b/c/d/e"), Path("a/b/c/d/e/f")]
    os.makedirs(tmp_path / dirs[-1])
    archive_paths = []
    dataset_paths = []

    data = np.arange(100).reshape(10, 10)
    for dir_idx, d in enumerate(dirs):
        n5_path = tmp_path / d / f"testn5_{dir_idx}.n5"
        archive_paths.append(n5_path)
        with z5py.File(n5_path.as_posix(), use_zarr_format=False) as f:
            for dir_idx_2, d2 in enumerate(dirs):
                inner_path = d2 / f"dataset{dir_idx_2}"
                dataset_paths.append(n5_path / inner_path)
                ds = f.create_dataset(inner_path.as_posix(), shape=data.shape, dtype=data.dtype)
                ds[...] = data
                ds.attrs["axes"] = "yx"

    file_glob = tmp_path / "**/testn5_*.n5"
    globbed = DataSourceUrl.glob_fs_path(file_glob)
    for p in archive_paths:
        assert p in globbed
    assert len(globbed) == len(archive_paths)

    dataset_glob = tmp_path / "**/testn5_*.n5/**/dataset*"
    globbed_dataset_paths = DataSourceUrl.glob_archive_path(dataset_glob)
    assert globbed_dataset_paths == dataset_paths
    urls = DataSourceUrl.glob(dataset_glob.as_posix())
    assert urls == [p.as_posix() for p in globbed_dataset_paths]
