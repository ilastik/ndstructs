from ndstructs.datasource.n5_attributes import N5DatasetAttributes
import re
from pathlib import Path
import json
from fs.base import FS as FileSystem

from ndstructs.array5D import Array5D
from ndstructs.datasource.DataSource import UnsupportedUrlException
from ndstructs.datasource.N5DataSource import N5Block
from ndstructs.datasink.DataSink import DataSink


class N5DatasetSink(DataSink):
    # @privatemethod
    def __init__(
        self,
        *,
        path: Path,  # dataset path, e.g. "mydata.n5/mydataset"
        filesystem: FileSystem,
        attributes: N5DatasetAttributes
    ):
        super().__init__(
            path=path,
            filesystem=filesystem,
            dtype=attributes.dataType,
            tile_shape=attributes.blockSize,
            interval=attributes.dimensions.to_interval5d(),
            location=attributes.location,
        )
        self.attributes = attributes

    @classmethod
    def create(
        cls,
        *,
        path: Path,  # dataset path, e.g. "mydata.n5/mydataset"
        filesystem: FileSystem,
        attributes: N5DatasetAttributes,
    ) -> "N5DatasetSink":
        match = re.search(r"^(?P<outer_path>.+\.n5/)(?P<inner_path>.+)$", path.as_posix(), re.IGNORECASE)
        if not match:
            raise UnsupportedUrlException(path)

        outer_path : Path = Path(match.group("outer_path"))
        inner_path : Path = Path(match.group("inner_path"))
        full_path = outer_path / inner_path

        filesystem.makedirs(full_path.as_posix(), recreate=True)

        with filesystem.openbin(outer_path.joinpath("attributes.json").as_posix(), "w") as f:
            f.write(json.dumps({"n5": "2.0.0"}).encode("utf8"))

        with filesystem.openbin(full_path.joinpath("attributes.json").as_posix(), "w") as f:
            f.write(json.dumps(attributes.to_json_data()).encode("utf-8"))

        # create all directories in the constructor to avoid races when processing tiles
        created_dirs = set()
        for tile in attributes.interval.split(attributes.blockSize):
            dir_path = path / attributes.get_tile_path(tile).parent
            if dir_path and dir_path not in created_dirs:
                print(f"Will create dir at {dir_path}")
                filesystem.makedirs(dir_path.as_posix())
                created_dirs.add(dir_path)

        return N5DatasetSink(
            path=path,
            filesystem=filesystem,
            attributes=attributes,
        )

    @classmethod
    def open(cls, *, path: Path, filesystem: FileSystem) -> "N5DatasetSink":
        with filesystem.openbin(path.joinpath("attributes.json").as_posix(), "r") as f:
            attributes_json = f.read().decode("utf8")
        attributes = N5DatasetAttributes.from_json_data(json.loads(attributes_json))
        return N5DatasetSink(filesystem=filesystem, path=path, attributes=attributes)

    def write(self, data: Array5D) -> None:
        tile = N5Block.fromArray5D(data)
        tile_path = self.path / self.attributes.get_tile_path(data.interval)
        with self.filesystem.openbin(tile_path.as_posix(), "w") as f:
            f.write(tile.to_n5_bytes(axiskeys=self.attributes.axiskeys, compression=self.attributes.compression))
