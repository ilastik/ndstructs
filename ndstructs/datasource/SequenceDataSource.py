from typing import List, Optional

from ndstructs.datasource import DataSource
from ndstructs import Shape5D, Point5D, Array5D


class SequenceDataSource(DataSource):
    def __init__(
        self,
        urls: List[str],
        *,
        stack_axis: str,
        datasources: List[DataSource] = None,
        tile_shape_hint: Optional[Shape5D] = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        self.urls = urls
        self.stack_axis = stack_axis
        if datasources:
            self._datasources = datasources
        else:
            self._datasources = [DataSource.create(url, tile_shape_hint=tile_shape_hint) for url in urls]
        if len(set(ds.full_shape.with_coord(**{stack_axis: 1}) for ds in self._datasources)) > 1:
            raise ValueError("Provided files have different dimensions on the non-stacking axis")
        if any(ds.dtype != self._datasources[0].dtype for ds in self._datasources):
            raise ValueError("All datasources must have the same data type!")
        self._stack_slices: List[slice] = []
        offset = Point5D.zero()
        for ds in self._datasources:
            self._stack_slices.append(ds.full_roi.translated(offset)[stack_axis])
            offset += Point5D.zero(**{stack_axis: ds.shape[stack_axis]})
        super().__init__(url=":".join(urls), t=t, c=c, x=x, y=y, z=z)

    @property
    def full_shape(self) -> Shape5D:
        stack_size = sum(ds.shape[self.stack_axis] for ds in self._datasources)
        return self._datasources[0].shape.with_coord(**{self.stack_axis: stack_size})

    def rebuild(
        self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)
    ) -> "SequenceDataSource":
        return SequenceDataSource(
            urls=self.urls, stack_axis=self.stack_axis, datasources=self._datasources, t=t, c=c, x=x, y=y, z=z
        )

    @property
    def dtype(self):
        return self._datasources[0].dtype

    def get(self):
        start = self.roi.start[self.stack_axis]
        stop = self.roi.stop[self.stack_axis]

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

        out = self._allocate(fill_value=0)
        for ds, stack_slice in zip(self._datasources[middle:], self._stack_slices[middle:]):
            if stack_slice.start > stop:
                break
            offset = Point5D.zero(**{self.stack_axis: stack_slice.start})
            local_roi = self.roi.translated(-offset).clamped(ds.full_roi)
            stack_slice_data = ds.resize(local_roi).retrieve().translated(offset)
            out.set(stack_slice_data, autocrop=True)

        return out

    @property
    def tile_shape(self) -> Shape5D:
        # FIXME: does this make sense?
        return self._datasources[0].tile_shape
