import numpy


class DepthmapCleaner:
    def __init__(self) -> None:
        ...

    def add_view(
        self,
        arg0: numpy.ndarray[float64],
        arg1: numpy.ndarray[float64],
        arg2: numpy.ndarray[float64],
        arg3: numpy.ndarray[float32],
    ) -> None:
        ...

    def clean(self) -> object:
        ...

    def set_min_consistent_views(self, arg0: int) -> None:
        ...

    def set_same_depth_threshold(self, arg0: float) -> None:
        ...


class DepthmapEstimator:
    def __init__(self) -> None:
        ...

    def add_view(
        self,
        arg0: numpy.ndarray[float64],
        arg1: numpy.ndarray[float64],
        arg2: numpy.ndarray[float64],
        arg3: numpy.ndarray[uint8],
        arg4: numpy.ndarray[uint8],
    ) -> None:
        ...

    def compute_brute_force(self) -> object:
        ...

    def compute_patch_match(self) -> object:
        ...

    def compute_patch_match_sample(self) -> object:
        ...

    def set_depth_range(self, arg0: float, arg1: float, arg2: int) -> None:
        ...

    def set_min_patch_sd(self, arg0: float) -> None:
        ...

    def set_patch_size(self, arg0: int) -> None:
        ...

    def set_patchmatch_iterations(self, arg0: int) -> None:
        ...


class DepthmapPruner:
    def __init__(self) -> None:
        ...

    def add_view(
        self,
        arg0: numpy.ndarray[float64],
        arg1: numpy.ndarray[float64],
        arg2: numpy.ndarray[float64],
        arg3: numpy.ndarray[float32],
        arg4: numpy.ndarray[float32],
        arg5: numpy.ndarray[uint8],
        arg6: numpy.ndarray[uint8],
        arg7: numpy.ndarray[uint8],
    ) -> None:
        ...

    def prune(self) -> object:
        ...

    def set_same_depth_threshold(self, arg0: float) -> None:
        ...


class OpenMVSExporter:
    def __init__(self) -> None:
        ...

    def add_camera(self, arg0: str, arg1: numpy.ndarray[float64], arg2: int, arg3: int) -> None:
        ...

    def add_point(self, arg0: numpy.ndarray[float64], arg1: list) -> None:
        ...

    def add_shot(
        self,
        arg0: str,
        arg1: str,
        arg2: str,
        arg3: numpy.ndarray[float64],
        arg4: numpy.ndarray[float64],
    ) -> None:
        ...

    def export(self, arg0: str) -> None:
        ...