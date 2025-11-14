import os
import shutil

import numpy as np
import skimage
import zarr
import zarr.storage
from ome_zarr_models._v06.collection import Collection
from ome_zarr_models._v06.coordinate_transforms import (
    Axis,
    CoordinateSystem,
    Identity,
    Rotation,
    Transform,
    Translation,
)
from ome_zarr_models._v06.image import Image
from pydantic_zarr.v3 import ArraySpec
from rich.pretty import pprint

name = "example1"
path = "./examples/example1.zarr"

data = skimage.data.cells3d()
data.shape

dz = 0.5
dy = 0.2
dx = 0.2
pixel_unit = "um"
ds_factor = 2


arr0 = data
arr1 = skimage.transform.downscale_local_mean(arr0, (1, 1, ds_factor, ds_factor))
arr2 = skimage.transform.downscale_local_mean(arr1, (1, 1, ds_factor, ds_factor))
array_specs = [
    ArraySpec.from_array(arr0),
    ArraySpec.from_array(arr1),
    ArraySpec.from_array(arr2),
]


physical_coord_system = CoordinateSystem(
    name="physical",
    axes=(
        Axis(name="z", type="space", unit=pixel_unit),
        Axis(name="c", type="channel", discrete=True),
        Axis(name="y", type="space", unit=pixel_unit),
        Axis(name="x", type="space", unit=pixel_unit),
    ),
)
world_coord_system = CoordinateSystem(
    name="world",
    axes=(
        Axis(name="z", type="space", unit="um"),
        Axis(name="c", type="channel", discrete=True),
        Axis(name="y", type="space", unit="um"),
        Axis(name="x", type="space", unit="um"),
    ),
)


# Create a rotation matrix for 45 degrees around z-axis
# The axes are: z, c, y, x (indices 0, 1, 2, 3)
# We want to rotate in the y-x plane, leaving z and c unchanged
angle = np.radians(45)
cos_a = np.cos(angle)
sin_a = np.sin(angle)

# 4x4 rotation matrix: identity for z and c, rotation in y-x plane
rotation_matrix = [
    [1, 0, 0, 0],  # z unchanged
    [0, 1, 0, 0],  # c (channel) unchanged
    [0, 0, cos_a, -sin_a],  # y' = cos(θ)*y - sin(θ)*x
    [0, 0, sin_a, cos_a],  # x' = sin(θ)*y + cos(θ)*x
]

transform_physical_to_world = Rotation(
    rotation=rotation_matrix,
    input="physical",
    output="world",
)

ome_zarr_image = Image.new(
    array_specs=array_specs,
    paths=[f"level{i}" for i in range(len(array_specs))],
    scales=[
        [dz * ds_factor**i, 1, dy * ds_factor**i, dx * ds_factor**i]
        for i in range(len(array_specs))
    ],
    translations=[
        [0, 0, dy * (ds_factor**i - 1) / 2, dx * (ds_factor**i - 1) / 2]
        for i in range(len(array_specs))
    ],
    physical_coord_system=physical_coord_system,
    name="Example_Image",
    coord_transforms=[transform_physical_to_world],
    coord_systems=[world_coord_system],
)

if os.path.exists(path):
    shutil.rmtree(path)
store = zarr.storage.LocalStore(path)
ome_zarr_image.to_zarr(store=store, path="/")


# Write the numpy array data to the zarr store
group = zarr.open_group(store=store, mode="a")
group["level0"][:] = arr0
group["level1"][:] = arr1
group["level2"][:] = arr2
