from typing import List, Optional, Union, Sequence
from pathlib import Path
import bisect
from fs.base import FS
import itertools

from ndstructs.datasource.DataSource import DataSource
from ndstructs.datasource.DataRoi import DataRoi
from ndstructs import Shape5D, Interval5D, Point5D, Array5D


class SequenceDataSource(DataSource):
    def __init__(
        self,
        paths: List[Path],
        *,
        stack_axis: str,
        layer_axiskeys: Union[str, Sequence[str]] = "",
        location: Point5D = Point5D.zero(),
        filesystems: Sequence[FS] = (),
    ):
        layer_axiskeys = layer_axiskeys or [""] * len(paths)
        assert len(layer_axiskeys) == len(paths)
        self.stack_axis = stack_axis
        self.layers: List[DataSource] = []
        self.layer_offsets: List[int] = []
        layer_offset = Point5D.zero()
        for layer_path, layer_fs in itertools.zip_longest(paths, filesystems):
            layer = DataSource.create(layer_path, location=layer_offset, filesystem=layer_fs)
            self.layers.append(layer)
            self.layer_offsets.append(layer_offset[stack_axis])
            layer_offset += Point5D.zero(**{stack_axis: layer.shape[stack_axis]})

        if len(set(layer.shape.updated(**{stack_axis: 1}) for layer in self.layers)) > 1:
            raise ValueError("Provided files have different dimensions on the non-stacking axis")
        if any(layer.dtype != self.layers[0].dtype for layer in self.layers):
            raise ValueError("All layers must have the same data type!")

        stack_size = sum(layer.shape[self.stack_axis] for layer in self.layers)
        full_shape = self.layers[0].shape.updated(**{self.stack_axis: stack_size})

        super().__init__(
            url=":".join(p.as_posix() for p in paths),
            shape=full_shape,
            name="Stack from " + ":".join(p.name for p in paths),
            dtype=self.layers[0].dtype,
            location=location,
            axiskeys=stack_axis + Point5D.LABELS.replace(stack_axis, ""),
        )

    def _get_tile(self, tile: Interval5D) -> Array5D:
        first_layer_idx = bisect.bisect_left(self.layer_offsets, tile.start[self.stack_axis])
        out = self._allocate(interval=tile, fill_value=0)
        for layer, layer_offset in zip(self.layers[first_layer_idx:], self.layer_offsets[first_layer_idx:]):
            if layer_offset > tile.stop[self.stack_axis]:
                break
            layer_tile = tile.clamped(layer.interval)
            layer_data = layer.retrieve(layer_tile)
            out.set(layer_data, autocrop=True)

        return out
