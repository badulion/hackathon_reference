from typing import Tuple, Literal, Iterable
import numpy.typing as npt
import numpy as np
from .dataclasses import SimulationRawData


class Crop:
    """
    Class for cropping the simulation data.

    Parameters
    ----------
    crop_size : Tuple[int, int, int]
        Size of the resulting data

    crop_position : Literal['random', 'center']
        Position of the crop
    """

    def __init__(self, 
                 crop_size: Tuple[int, int, int]):
        super().__init__()

        if not isinstance(crop_size, Iterable):
            raise ValueError("Crop size should be a tuple")
        elif len(crop_size) != 3:
            raise ValueError("Crop size should have 3 dimensional")
        elif not all((isinstance(i, int) for i in crop_size)):
            raise ValueError("Crop size should contain only integers")
        elif not (np.array(crop_size) > 0).all():
            raise ValueError("Crop size should be larger than 0")
        

        self.crop_size = crop_size

    def __call__(self, simulation_raw_data: SimulationRawData):
        """
        Crops data based on the given arguments. In the case of the `center` crop position make almost equal margins on both sides.
        In the case of the `random` crop position, the crop margin is randomly selected as well as crop start. 
        """
        crop_size = self.crop_size
        full_size = simulation_raw_data.properties.shape[1:]
        crop_start = self._sample_crop_start(full_size, crop_size)
        crop_mask = tuple(slice(crop_start[i], crop_start[i] + crop_size[i]) for i in range(3))

        return SimulationRawData(
            simulation_name=simulation_raw_data.simulation_name,
            properties=self._crop_array(simulation_raw_data.properties, crop_mask, 1),
            field=self._crop_array(simulation_raw_data.field, crop_mask, 3),
            subject=self._crop_array(simulation_raw_data.subject, crop_mask, 0),
            coil=self._crop_array(simulation_raw_data.coil, crop_mask, 0),
        )
    
    def _crop_array(self, 
                    array: npt.NDArray[np.float32],
                    crop_mask: Tuple[slice, slice, slice], 
                    starting_axis: int) -> npt.NDArray[np.float32]:
        crop_mask = (slice(None), )*starting_axis + crop_mask
        return array[*crop_mask]

    
    def _sample_crop_start(self, full_size: Tuple[int, int, int], crop_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        for i in range(3):
            if crop_size[i] > full_size[i]:
                raise ValueError(f"crop size {crop_size} is larger than full size {full_size}")
        if full_size == crop_size:
            return (0, 0, 0)    
        else:
            crop_start = [(full_size[i] - crop_size[i]) // 2 for i in range(3)]
        return crop_start
    
    
class CropOnlySubject:
    def __init__(self):
        pass
    
    def _calculate_crop_indices(self, 
                                subject_mask: npt.NDArray[np.float32]) -> Tuple[int, int, int]:
        mask_x = np.any(subject_mask, axis=(1, 2))
        mask_y = np.any(subject_mask, axis=(0, 2))
        mask_z = np.any(subject_mask, axis=(0, 1))
        x_min, x_max = np.where(mask_x)[0][[0, -1]]
        y_min, y_max = np.where(mask_y)[0][[0, -1]]
        z_min, z_max = np.where(mask_z)[0][[0, -1]]
        return x_min, x_max, y_min, y_max, z_min, z_max
    
    def __call__(self, simulation_raw_data: SimulationRawData) -> SimulationRawData:
        x_min, x_max, y_min, y_max, z_min, z_max = self._calculate_crop_indices(simulation_raw_data.subject)
        crop_mask = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
        return SimulationRawData(
            simulation_name=simulation_raw_data.simulation_name,
            properties=self._crop_array(simulation_raw_data.properties, crop_mask, 1),
            field=self._crop_array(simulation_raw_data.field, crop_mask, 3),
            subject=self._crop_array(simulation_raw_data.subject, crop_mask, 0),
            coil=self._crop_array(simulation_raw_data.coil, crop_mask, 0),
        )
        
    def _crop_array(self, 
                    array: npt.NDArray[np.float32],
                    crop_mask: Tuple[slice, slice, slice], 
                    starting_axis: int) -> npt.NDArray[np.float32]:
        crop_mask = (slice(None), )*starting_axis + crop_mask
        return array[*crop_mask]