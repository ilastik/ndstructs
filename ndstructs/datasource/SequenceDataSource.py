from typing import List, Optional, Union
from pathlib import Path

from ndstructs.datasource import DataSource, DataSourceUrl
from ndstructs.datasource import BackedSlice5D
from ndstructs import Shape5D, Slice5D, Point5D, Array5D


class SequenceDataSource(DataSource):
    def __init__(
        self, url: Union[Path, str], *, stack_axis: str, tile_shape: Optional[Shape5D] = None, slice_axiskeys: str = ""
    ):
        self.stack_axis = stack_axis
        urls = DataSourceUrl.glob(Path(url))
        self._datasources = [DataSource.create(url, tile_shape=tile_shape, axiskeys=slice_axiskeys) for url in urls]

        if len(set(ds.shape.with_coord(**{stack_axis: 1}) for ds in self._datasources)) > 1:
            raise ValueError("Provided files have different dimensions on the non-stacking axis")
        if any(ds.dtype != self._datasources[0].dtype for ds in self._datasources):
            raise ValueError("All datasources must have the same data type!")
        self._stack_slices: List[slice] = []
        offset = Point5D.zero()
        for ds in self._datasources:
            self._stack_slices.append(ds.roi.translated(offset)[stack_axis])
            offset += Point5D.zero(**{stack_axis: ds.shape[stack_axis]})

        stack_size = sum(ds.shape[self.stack_axis] for ds in self._datasources)
        full_shape = self._datasources[0].shape.with_coord(**{self.stack_axis: stack_size})

        super().__init__(
            url=url,
            tile_shape=tile_shape or self._datasources[0].tile_shape,  # FIXME is this always safe?
            shape=full_shape,
            name="Stack from {url}",
            dtype=self._datasources[0].dtype,
            axiskeys=stack_axis + slice_axiskeys,
        )

    def _get_tile(self, tile: Slice5D):
        start = tile.start[self.stack_axis]
        stop = tile.stop[self.stack_axis]

        left = 0
        right = len(self._stack_slices) - 1

        while left <= right:
            middle = (left + right) // 2
            slc = self._stack_slices[middle]
            if start < slc.start:
                right = middle - 1
            elif start > slc.stop:
                left = middle + 1
            else:
                break
        else:
            raise RuntimeError(f"Could not find a slice in the stack that contained {self.roi}")

        out = self._allocate(slc=tile, fill_value=0)
        for ds, stack_slice in zip(self._datasources[middle:], self._stack_slices[middle:]):
            if stack_slice.start > stop:
                break
            offset = Point5D.zero(**{self.stack_axis: stack_slice.start})
            local_roi = self.roi.translated(-offset).clamped(ds.roi)
            stack_slice_data = BackedSlice5D(ds, **local_roi.to_dict()).retrieve().translated(offset)
            out.set(stack_slice_data, autocrop=True)

        return out
