from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    List,
    Self,
    Iterable,
    Callable,
    TypeAlias,
)
import os
from dataclasses import dataclass
from pathlib import Path
from enum import Enum, Flag, auto
from copy import deepcopy
from functools import partial

import yaml
from omegaconf import OmegaConf, DictConfig
import hydra_zen as zen


def partial_representer(dumper, data):
    func_name = f"{data.func.__module__}.{data.func.__name__}"
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map", {"_target_": func_name, "_partial_": True}
    )


yaml.add_representer(partial, partial_representer)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def parent(
    key: Optional[str] = None, /, *, depth: int = 0, _parent_: DictConfig
) -> Any:
    """This method will recursively search up the parent chain for the key and return
    the parent key. If key is not specified, it will return the parent's key.

    For instance, a heavily nested value might want to access a value some level
    higher but it may be hazardous to use relative paths (i.e. ${..key}) since
    the config may be changed. Instead, we'll search up for a specific key to set the
    value to. Helpful for setting unique names for an object in a nested config.

    NOTE: This technically uses hidden attributes (i.e. _parent).

    Args:
        key (Optional[str]): The key to search for. If None, will return the parent's
            key. Defaults to None.
        depth (int, optional): The depth of the search. Used internally
            in this method and unsettable from the config. Avoids checking the parent
            key.
        _parent_ (DictConfig): The parent config to search in.
    """
    if _parent_ is None:
        # Parent will be None if we're at the top level
        raise ValueError(f"Key {key} not found in parent chain.")

    if key is None:
        # If the key is None, we'll return the parent's key
        assert _parent_._key() is not None, "Parent key is None."
        return _parent_._key()

    if depth != 0 and isinstance(_parent_, DictConfig) and key in _parent_:
        # If we're at a key that's not the parent and the parent has the key we're
        # looking for, we'll return the parent
        return parent(None, depth=depth + 1, _parent_=_parent_)
    else:
        # Otherwise, we'll keep searching up the parent chain
        return parent(key, depth=depth + 1, _parent_=_parent_._parent)


OmegaConf.register_new_resolver("parent", parent, replace=True)


def config_wrapper(cls=None, /, dataclass_kwargs: Dict[str, Any] | None = ...):
    """This is a wrapper of the dataclass decorator that adds the class to the hydra
    store.

    The hydra store is used to construct structured configs from the yaml files.
    NOTE: Only some primitive datatypes are supported by Hydra/OmegaConf.

    Args:
        dataclass_kwargs (Dict[str, Any] | None): The kwargs to pass to the dataclass
            decorator. If unset, will use the defaults. If set to None, the class
            will not be wrapped as a dataclass.
    """

    # Update the kwargs for the dataclass with some defaults
    default_dataclass_kwargs = dict(repr=False, slots=True, eq=False, kw_only=True)
    if dataclass_kwargs is ...:
        # Set to the default dataclass kwargs
        dataclass_kwargs = default_dataclass_kwargs
    elif isinstance(dataclass_kwargs, dict):
        # Update the default dataclass kwargs with the given dataclass kwargs
        dataclass_kwargs = {**default_dataclass_kwargs, **dataclass_kwargs}

    def wrapper(cls):
        if dataclass_kwargs is not None:
            cls = dataclass(cls, **dataclass_kwargs)

        # Add to the hydra store
        if (None, cls.__name__) not in zen.store:
            zen.store(
                zen.builds(cls, populate_full_signature=True, hydra_convert="partial"),
                name=cls.__name__,
                zen_dataclass={**dataclass_kwargs, "cls_name": cls.__name__},
                builds_bases=(cls,),
            )

        return cls

    if cls is None:
        return wrapper
    return wrapper(cls)


@config_wrapper
class MjCambrianBaseConfig:
    """Base config for all configs. This is an abstract class.

    Attributes:
        custom (Optional[Dict[Any, str]]): Custom data to use. This is useful for
            code-specific logic (i.e. not in yaml files) where you want to store
            data that is not necessarily defined in the config. It's ignored by
            OmegaConf.
    """

    custom: Optional[Dict[Any, str]] = None

    def save(self, path: Path | str):
        """Save the config to a yaml file."""

        class CustomDumper(yaml.CSafeDumper):
            pass

        # Add the following resolver temporarily to the yaml file
        def str_representer(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        CustomDumper.add_representer(str, str_representer)

        with open(path, "w") as f:
            f.write(
                yaml.dump(
                    yaml.safe_load(OmegaConf.to_yaml(self)),
                    sort_keys=False,
                    Dumper=CustomDumper,
                )
            )

    def copy(self) -> Self:
        """Copy the config such that it is a new instance."""
        return deepcopy(self)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of the key. Like `dict.get`."""
        return getattr(self, key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        """Assign the default value to the key if it is not already set. Like
        `dict.setdefault`."""
        if not hasattr(self, key) or getattr(self, key) is None:
            setattr(self, key, default)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if the dataclass contains a key with the given name."""
        return key in self.__annotations__

    def __str__(self):
        return OmegaConf.to_yaml(self)


MjCambrianXMLConfig: TypeAlias = Any
"""Actual type: List[Dict[str, Self]]

The actual type is a nested list of dictionaries that's recursively defined. We use a
list here because then we can have non-unique keys.

This defines a custom xml config. This can be used to define custom xmls which 
are built from during the initialization phase of the environment. The config is 
structured as follows:

```yaml
parent_key1:
    - child_key1: 
        - attr1: val1
        - attr2: val2
    - child_key2:
        - attr1: val1
        - attr2: val2
child_key1:
    - child_key2:
        - attr1: ${parent_key1.child_key1.attr2}
- child_key2:
    - child_key3: ${parent_key1.child_key1}
```

which will construct an xml that looks like:

```xml
<parent_key1>
    <child_key1 attr1="val1" attr2="val2">
        <child_key2 attr1="val2"/>
    </child_key1>
    <child_key2>
        <attr1>val1</attr1>
        <attr2>val2</attr2>
        <child_key3 attr1="val1" attr2="val2">
    </child_key2>
</parent_key1>
```

This is a verbose representation for xml files. This is done
to allow interpolation through hydra/omegaconf in the xml files and without the need
for a complex xml parser omegaconf resolver.

TODO: I think this type (minus the Self) is supported as of OmegaConf issue #890.
"""

MjCambrianActivationFn: TypeAlias = Any
"""Actual type: torch.nn.Module"""

MjCambrianRewardFn: TypeAlias = Any
"""Actual type: Callable[[MjCambrianAnimal, Dict[str, Any], ...], float]"""


@config_wrapper
class MjCambrianGenerationConfig(MjCambrianBaseConfig):
    """Config for a generation. Used for type hinting.

    Attributes:
        rank (int): The rank of the generation. A rank is a unique identifier assigned
            to each process, where a processes is an individual evo runner running on a
            separate computer. In the context of a cluster, each node that is running
            an evo job is considered one rank, where the rank number is a unique int.
        generation (int): The generation number. This is used to uniquely identify the
            generation.
    """

    rank: int
    generation: int

    def to_path(self) -> Path:
        return Path(f"generation_{self.generation}") / f"rank_{self.rank}"


@config_wrapper
class MjCambrianTrainingConfig(MjCambrianBaseConfig):
    """Settings for the training process. Used for type hinting.

    Attributes:
        total_timesteps (int): The total number of timesteps to train for.
        max_episode_steps (int): The maximum number of steps per episode.
        n_steps (int): The number of steps to take per training batch.

        learning_rate (float): The learning rate to use for training. NOTE: sb3 default
            is 3e-4.
        n_epochs (int): The number of epochs to use for training. NOTE: sb3 default is
            10.
        gae_lambda (float): The lambda value to use for the generalized advantage
            estimation. NOTE: sb3 default is 0.95.
        batch_size (int): The batch size to use for training.
        n_envs (int): The number of environments to use for training.

        eval_freq (int): The frequency at which to evaluate the model.
        no_improvement_reward_threshold (float): The reward threshold at which to stop
            training.
        max_no_improvement_evals (int): The maximum number of evals to take without
            improvement before stopping training.
        min_no_improvement_evals (int): The minimum number of evaluations to perform
            before stopping training if max_no_improvement_steps is reached.

        model (MjCambrianModelConfig): The settings for the model.
    """

    total_timesteps: int
    max_episode_steps: int
    n_steps: int

    learning_rate: float
    n_epochs: int
    gae_lambda: float
    batch_size: int
    n_envs: int

    eval_freq: int
    no_improvement_reward_threshold: float
    max_no_improvement_evals: int
    min_no_improvement_evals: int

    @config_wrapper
    class MjCambrianModelConfig(MjCambrianBaseConfig):
        """Settings for the model. Used for type hinting.

        Attributes:
            checkpoint_path (Optional[str]): The path to the model checkpoint to
                load. If None, training will start from scratch.
            policy_path (Optional[str]): The path to the policy checkpoint to load.
                Should be a `.pt` file that was saved using MjCambrianModel.save_policy.

            features_extractor_activation (MjCambrianActivationFn): The activation
                function to use for the features extractor. Should be a nn.Module
        """

        checkpoint_path: Optional[str] = None
        policy_path: Optional[str] = None

        features_extractor_activation: MjCambrianActivationFn

    model: MjCambrianModelConfig


@config_wrapper
class MjCambrianMazeConfig(MjCambrianBaseConfig):
    """Defines a map config. Used for type hinting.

    Attributes:
        ref (Optional[str]): Reference to a named maze config. Used to share walls and
            other geometries/assets. A check will be done to ensure the walls are
            identical between configs.

        map (List[List[str]]): The map to use for the maze. It's a 2D array where
            each element is a string and corresponds to a "pixel" in the map. See
            `maze.py` for info on what different strings mean.
        xml (str): The xml for the maze. This is the xml that will be used to
            create the maze.

        difficulty (float): The difficulty of the maze. This is used to determine
            the selection probability of the maze when the mode is set to "DIFFICULTY".
            The value should be set between 0 and 100, where 0 is the easiest and 100
            is the hardest.

        size_scaling (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.
        flip (bool): Whether to flip the maze or not. If True, the maze will be
            flipped along the x-axis.
        smooth_walls (bool): Whether to smooth the walls such that they are continuous
            appearing. This is an approximated as a spline fit to the walls.

        hide_targets (bool): Whether to hide the target or not. If True, the target
            will be hidden.
        use_target_light_sources (bool): Whether to use a target light sources or not.
            If False, the colored target sites will be used (e.g. a red sphere).
            Otherwise, a light source will be used. The light source is simply a spot
            light facing down.

        wall_texture_map (Dict[str, List[str]]): The mapping from texture id to
            texture names. Textures in the list are chosen at random. If the list is of
            length 1, only one texture will be used. A length >= 1 is required.
            The keyword "default" is required for walls denoted simply as 1 or W.
            Other walls are specified as 1/W:<texture id>.

        init_goal_pos (Optional[Tuple[float, float]]): The initial position of the
            goal in the maze. If unset, will be randomly generated.
        eval_goal_pos (Optional[Tuple[float, float]]): The evaluation position of the
            goal in the maze. If unset, will be randomly generated.

        use_adversary (bool): Whether to use an adversarial target or not. If
            True, a second target will be created which is deemed adversarial. Also,
            the target's will be given high frequency textures which correspond to
            whether a target is adversarial or the true goal. This is done in hopes of
            having the animal learn to see high frequency input.
        init_adversary_pos (Optional[Tuple[float, float]]): The initial position
            of the adversary target in the maze. If unset, will be randomly generated.
        eval_adversary_pos (Optional[Tuple[float, float]]): The evaluation
            position of the adversary target in the maze. If unset, will be randomly
            generated.
    """

    ref: Optional[str] = None

    map: Optional[str] = None
    xml: MjCambrianXMLConfig

    difficulty: float

    size_scaling: float
    height: float
    flip: bool
    smooth_walls: bool

    hide_targets: bool
    use_target_light_sources: bool

    wall_texture_map: Dict[str, List[str]]

    init_goal_pos: Optional[Tuple[float, float]] = None
    eval_goal_pos: Optional[Tuple[float, float]] = None

    use_adversary: bool
    init_adversary_pos: Optional[Tuple[float, float]] = None
    eval_adversary_pos: Optional[Tuple[float, float]] = None


@config_wrapper
class MjCambrianCameraConfig(MjCambrianBaseConfig):
    """Defines a camera config. Used for type hinting. This is a wrapper of
    mj.mjvCamera that is used to configure the camera in the viewer.

    Attributes:
        type (Optional[int]): The type of camera.
        fixedcamid (Optional[int]): The id of the camera to use.
        trackbodyid (Optional[int]): The id of the body to track.

        lookat (Optional[Tuple[float, float, float]]): The point to look at.
        distance (Optional[float]): The distance from the camera to the lookat point.
        azimuth (Optional[float]): The azimuth angle.
        elevation (Optional[float]): The elevation angle.

        typename (Optional[str]): The type of camera as a string. Takes presidence over
            type. Converted to mjtCamera with mjCAMERA_{typename.upper()}.
        fixedcamname (Optional[str]): The name of the camera. Takes presidence over
            fixedcamid. Used to determine the fixedcamid using mj.mj_name2id.
        trackbodyname (Optional[str]): The name of the body to track. Takes presidence
            over trackbodyid. Used to determine the trackbodyid using mj.mj_name2id.

        distance_factor (Optional[float]): The distance factor. This is used to
            calculate the distance from the camera to the lookat point. If unset, no
            scaling will be applied.
    """

    type: Optional[int] = None
    fixedcamid: Optional[int] = None
    trackbodyid: Optional[int] = None

    lookat: Optional[Tuple[float, float, float]] = None
    distance: Optional[float] = None
    azimuth: Optional[float] = None
    elevation: Optional[float] = None

    typename: Optional[str] = None
    fixedcamname: Optional[str] = None
    trackbodyname: Optional[str] = None

    distance_factor: Optional[float] = None


@config_wrapper
class MjCambrianRendererConfig(MjCambrianBaseConfig):
    """The config for the renderer. Used for type hinting.

    A renderer corresponds to a single camera. The renderer can then view the scene in
    different ways, like offscreen (rgb_array) or onscreen (human).

    Attributes:
        render_modes (List[str]): The render modes to use for the renderer. See
            `MjCambrianRenderer.metadata["render.modes"]` for options.

        maxgeom (Optional[int]): The maximum number of geoms to render.

        width (int): The width of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.
        height (int): The height of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.

        fullscreen (Optional[bool]): Whether to render in fullscreen or not. If True,
            the width and height are ignored and the window is rendered in fullscreen.
            This is only valid for onscreen renderers.

        camera_config (Optional[MjCambrianCameraConfig]): The camera config to use for
            the renderer.
        scene_options (Optional[Dict[str, Any]]): The scene options to use for the
            renderer. Keys are the name of the option as defined in MjvOption. For
            array options (like `flags`), the value should be another dict where the
            keys are the indices/mujoco enum keys and the values are the values to set.

        use_shared_context (bool): Whether to use a shared context or not.
            If True, the renderer will share a context with other renderers. This is
            useful for rendering multiple renderers at the same time. If False, the
            renderer will create its own context. This is computationally expensive if
            there are many renderers.
    """

    render_modes: List[str]

    maxgeom: Optional[int] = None

    width: Optional[int] = None
    height: Optional[int] = None

    fullscreen: Optional[bool] = None

    camera_config: Optional[MjCambrianCameraConfig] = None
    scene_options: Optional[Dict[str, Any]] = None

    use_shared_context: bool


@config_wrapper
class MjCambrianEyeConfig(MjCambrianBaseConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        name (Optional[str]): Placeholder for the name of the eye. If set, used
            directly. If unset, the name is set to `{animal.name}_eye_{eye_index}`.

        mode (str): The mode of the camera. Should always be "fixed". See the mujoco
            documentation for more info.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: width height.
        fov (Tuple[float, float]): Independent of the `fovy` field in the MJCF
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set. Fmt: fovx fovy.

        enable_optics (bool): Whether to enable optics or not.
        enable_aperture (bool): Whether to enable the aperture or not.
        enable_lens (bool): Whether to enable the lens or not.
        enable_phase_mask (bool): Whether to enable the phase mask or not.

        scene_angular_resolution: The angular resolution of the scene. This is used to
            determine the field of view of the scene. Specified in degrees.
        pixel_size: The pixel size of the sensor in meters.
        sensor_resolution (Tuple[int, int]): TODO
        add_noise (bool): TODO
        noise_std (float): TODO
        aperture_open (float): The aperture open value. This is the radius of the
            aperture. The aperture is a circle that is used to determine which light
            rays to let through. Only used if `enable_aperture` is True. Must be
            between 0 and 1.
        aperture_radius (float): The aperture radius value.
        wavelengths (Tuple[float, float, float]): The wavelengths to use for the
            intensity sensor. Fmt: wavelength_1 wavelength_2 wavelength_3
        depth_bins (int): The number of depth bins to use for the depth dependent psf.

        load_height_mask_from_file (bool): Whether to load the height mask from file or
            not. If True, the height mask will be loaded from the file specified in
            `height_mask_from_file`. If False, the psf wil be randomized or set to zeros
            using `randomize_psf_init`.
        height_mask_from_file (Optional[str]): The path to the height mask file to load.
        randomize_psf_init (bool): Whether to randomize the psf or not. If True, the psf
            will be randomized. If False, the psf will be set to zeros. Only used if
            `load_height_mask_from_file` is False.
        zernike_basis_path (Optional[str]): The path to the zernike basis file to load.
        psf_filter_size (Tuple[int, int]): The psf filter size. This is
            convolved across the image, so the actual resolution of the image is plus
            psf_filter_size / 2. Only used if `load_height_mask_from_file` is False.
            Otherwise the psf filter size is determined by the height mask.
        refractive_index (float): The refractive index of the eye.
        min_phi_defocus (float): TODO
        max_phi_defocus (float): TODO

        load_height_mask_from_file (bool): Whether to load the height mask from file or
            not. If True, the height mask will be loaded from the file specified in
            `height_mask_from_file`. If False, the psf wil be randomized or set to zeros
            using `randomize_psf_init`.
        height_mask_from_file (Optional[str]): The path to the height mask file to load.
        randomize_psf_init (bool): Whether to randomize the psf or not. If True, the psf
            will be randomized. If False, the psf will be set to zeros. Only used if
            `load_height_mask_from_file` is False.
        zernike_basis_path (Optional[str]): The path to the zernike basis file to load.

        psf_filter_size (Tuple[int, int]): The psf filter size. This is
            convolved across the image, so the actual resolution of the image is plus
            psf_filter_size / 2. Only used if `load_height_mask_from_file` is False.
            Otherwise the psf filter size is determined by the height mask.
        refractive_index (float): The refractive index of the eye.
        depth_bins (int): The number of depth bins to use for the depth sensor.
        min_phi_defocus (float): The minimum depth to use for the depth sensor.
        max_phi_defocus (float): The maximum depth to use for the depth sensor.
        wavelengths (Tuple[float, float, float]): The wavelengths to use for the
            intensity sensor. Fmt: wavelength_1 wavelength_2 wavelength_3
        #### Optics Params

        pos (Optional[Tuple[float, float, float]]): The initial position of the camera.
            Fmt: xyz
        quat (Optional[Tuple[float, float, float, float]]): The initial rotation of the
            camera. Fmt: wxyz.
        fovy (Optional[float]): The vertical field of view of the camera.
        focal (Optional[Tuple[float, float]]): The focal length of the camera.
            Fmt: focal_x focal_y.
        sensorsize (Optional[Tuple[float, float]]): The sensor size of the camera.
            Fmt: sensor_x sensor_y.

        coord (Optional[Tuple[float, float]]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the animal.
            Specified in degrees. Mutually exclusive with `pos` and `quat`. This attr
            isn't actually used by eye, but by the animal. The eye has no knowledge
            of the geometry it's trying to be placed on. Fmt: lat lon

        renderer_config (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + int(psf_filter_size/2)) of the eye.
    """

    name: Optional[str] = None

    mode: str
    resolution: Tuple[int, int]
    fov: Tuple[float, float]

    enable_optics: bool
    enable_aperture: bool
    enable_lens: bool
    enable_phase_mask: bool

    scene_resolution: Tuple[int, int]
    scene_angular_resolution: float
    pixel_size: float
    sensor_resolution: Tuple[int, int]
    add_noise: bool
    noise_std: float
    aperture_open: float
    aperture_radius: float
    wavelengths: Tuple[float, float, float]
    depth_bins: int

    load_height_mask_from_file: bool
    height_mask_from_file: Optional[str] = None
    randomize_psf_init: bool
    zernike_basis_path: Optional[str] = None
    psf_filter_size: Tuple[int, int]
    refractive_index: float
    min_phi_defocus: float
    max_phi_defocus: float

    pos: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    fovy: Optional[float] = None
    focal: Optional[Tuple[float, float]] = None
    sensorsize: Optional[Tuple[float, float]] = None

    coord: Optional[Tuple[float, float]] = None

    renderer_config: MjCambrianRendererConfig

    def to_xml_kwargs(self) -> Dict[str, Any]:
        kwargs = dict()

        def set_if_not_none(key: str, val: Any):
            if val is not None:
                if isinstance(val, Iterable) and not isinstance(val, str):
                    val = " ".join(map(str, val))
                kwargs[key] = val

        set_if_not_none("name", self.name)
        set_if_not_none("mode", self.mode)
        set_if_not_none("pos", self.pos)
        set_if_not_none("quat", self.quat)
        set_if_not_none("resolution", self.resolution)
        set_if_not_none("fovy", self.fovy)
        set_if_not_none("focal", self.focal)
        set_if_not_none("sensorsize", self.sensorsize)

        return kwargs


@config_wrapper
class MjCambrianAnimalModelConfig(MjCambrianBaseConfig):
    """Defines the config for an animal model. Used for type hinting.

    Attributes:
        xml (str): The xml for the animal model. This is the xml that will be used to
            create the animal model. You should use ${..name} to generate named
            attributes.
        body_name (str): The name of the body that defines the main body of the animal.
            This will probably be set through a MjCambrianAnimal subclass.
        joint_name (str): The root joint name for the animal. For positioning (see qpos)
            This will probably be set through a MjCambrianAnimal subclass.
        geom_name (str): The name of the geom that are used for eye placement.

        eyes_lat_range (Tuple[float, float]): The x range of the eye. This is used to
            determine the placement of the eye on the animal. Specified in degrees. This
            is the latitudinal/vertical range of the evenly placed eye about the
            animal's bounding sphere.
        eyes_lon_range (Tuple[float, float]): The y range of the eye. This is used to
            determine the placement of the eye on the animal. Specified in degrees. This
            is the longitudinal/horizontal range of the evenly placed eye about the
            animal's bounding sphere.
    """


@config_wrapper
class MjCambrianAnimalConfig(MjCambrianBaseConfig):
    """Defines the config for an animal. Used for type hinting.

    Attributes:
        init_pos (Tuple[float, float]): The initial position of the animal. If unset,
            the animal's position at each reset is generated randomly using the
            `maze.generate_reset_pos` method.
        constant_actions (Optional[List[float | None]]): The constant velocity to use for
            the animal. If not None, the len(constant_actions) must equal number of
            actuators defined in the model. For instance, if there are 3 actuators
            defined and it's desired to have the 2nd actuator be constant, then
            constant_actions = [None, 0, None]. If None, no constant action will be
            applied.

        use_intensity_obs (bool): Whether to use the intensity sensor observation.
        use_action_obs (bool): Whether to use the action observation or not.
        use_init_pos_obs (bool): Whether to use the initial position observation or not.
        use_current_pos_obs (bool): Whether to use the current position observation or
            not.
        n_temporal_obs (int): The number of temporal observations to use.

        eyes (List[MjCambrianEyeConfig]): The configs for the eyes.
            The key will be used as the default name for the eye, unless explicitly
            set in the eye config.
        intensity_sensor_config (Optional[MjCambrianEyeConfig]): The eye config to use
            for the intensity sensor. If unset, the intensity sensor will not be used.

        mutations_from_parent (Optional[List[str]]): The mutations applied to the child
            (this animal) from the parent. This is unused during mutation; it simply
            is a record of the mutations that were applied to the parent.
    """

    xml: MjCambrianXMLConfig

    body_name: str
    joint_name: str
    geom_name: str

    eyes_lat_range: Tuple[float, float]
    eyes_lon_range: Tuple[float, float]

    init_pos: Optional[Tuple[float, float]] = None
    constant_actions: Optional[List[float | None]] = None

    use_intensity_obs: bool
    use_action_obs: bool
    use_init_pos_obs: bool
    use_current_pos_obs: bool
    n_temporal_obs: int

    eyes: List[MjCambrianEyeConfig]
    intensity_sensor: Optional[MjCambrianEyeConfig] = None

    mutations_from_parent: Optional[List[str]] = None


@config_wrapper
class MjCambrianEnvConfig(MjCambrianBaseConfig):
    """Defines a config for the cambrian environment.

    Attributes:
        xml (MjCambrianXMLConfig): The xml for the scene. This is the xml that will be
            used to create the environment. See `MjCambrianXMLConfig` for more info.

        reward_fn (MjCambrianRewardFn): The reward function type to use. See the
            `MjCambrianRewardFn` for more info.

        use_goal_obs (bool): Whether to use the goal observation or not.
        terminate_at_goal (bool): Whether to terminate the episode when the animal
            reaches the goal or not.
        truncate_on_contact (bool): Whether to truncate the episode when the animal
            makes contact with an object or not.
        distance_to_target_threshold (float): The distance to the target at which the
            animal is assumed to be "at the target".
        action_penalty (float): The action penalty when it moves.
        adversary_penalty (float): The adversary penalty when it goes to the wrong target.
        contact_penalty (float): The contact penalty when it contacts the wall.
        force_exclusive_contact_penalty (bool): Whether to force exclusive contact
            penalty or not. If True, the contact penalty will be used exclusively for
            the reward. If False, the contact penalty will be used in addition to the
            calculated reward.

        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.

        use_renderer (bool): Whether to use the renderer. Should set to False if
            `render` will never be called. Defaults to True. This is useful to reduce
            the amount of vram consumed by non-rendering environments.
        add_overlays (bool): Whether to add overlays or not.
        clear_overlays_on_reset (bool): Whether to clear the overlays on reset or not.
            Consequence of setting to False is that if `add_position_tracking_overlay`
            is True and mazes change between evaluations, the sites will be drawn on top
            of each other which may not be desired. When record is False, the overlays
            are always cleared.
        add_position_tracking_overlay (bool): Whether to add the position
            tracking overlay which adds a site to the world at each position an
            animal has been.
        position_tracking_overlay_color (Optional[Tuple[float, float, float, float]]):
            The color of the position tracking overlay. Must be set if
            `add_position_tracking_overlay` is True. Fmt: rgba
        overlay_width (Optional[float]): The width of _each_ rendered overlay that's
            placed on the render output. This is primarily for debugging. If unset,
            no overlay will be added. This is a percentage!! It's the percentage of
            the total width of the render output.
        overlay_height (Optional[float]): The height of _each_ rendered overlay that's
            placed on the render output. This is primarily for debugging. If unset,
            no overlay will be added. This is a percentage!! It's the percentage of
            the total height of the render output.
        renderer_config (MjCambrianViewerConfig): The default viewer config to
            use for the mujoco viewer.

        maze_selection_criteria (Dict[str, Any]): The mode to use for choosing
            the maze. The `mode` key is required and must be set to a
            `MazeSelectionMode`. See `MazeSelectionMode` for other params or
            `maze.py` for more info.
        mazes (List[MjCambrianMazeConfig]): The configs for the mazes. Each
            maze will be loaded into the scene and the animal will be placed in a maze
            at each reset. The maze will be chosen based on the
            `maze_selection_criteria.mode` field.

        animals (List[MjCambrianAnimalConfig]): The configs for the animals.
            The key will be used as the default name for the animal, unless explicitly
            set in the animal config.

        eval_overrides (Optional[Dict[str, Any]]): Key/values to override the default
            env_config. Applied during evaluation only. Merged directly with the
            env_config. NOTE: Only some fields are actually used when loading.
            NOTE #2: Only the top level keys are used, as in this is a shallow
            merge.
    """

    xml: MjCambrianXMLConfig

    reward_fn: MjCambrianRewardFn

    use_goal_obs: bool
    terminate_at_goal: bool
    truncate_on_contact: bool
    distance_to_target_threshold: float
    action_penalty: float
    adversary_penalty: float
    contact_penalty: float
    force_exclusive_contact_penalty: bool

    frame_skip: int

    use_renderer: bool
    add_overlays: bool
    clear_overlays_on_reset: bool
    add_position_tracking_overlay: bool
    position_tracking_overlay_color: Optional[Tuple[float, float, float, float]] = None
    overlay_width: Optional[float] = None
    overlay_height: Optional[float] = None
    renderer_config: MjCambrianRendererConfig

    eval_overrides: Optional[Dict[str, Any]] = None

    class MazeSelectionMode(Enum):
        """The mode to use for choosing the maze. See `maze.py` for more info.

        NOTE: the `mode` key is required for the criteria dict. other keys are passed
        as kwargs to the selection method.

        Ex:
            # Choose a random maze
            maze_selection_criteria:
                mode: RANDOM

            # Choose a maze based on difficulty
            maze_selection_criteria:
                mode: DIFFICULTY
                schedule: logistic

            # From the command line
            -o ...maze_selection_criteria="{mode: DIFFICULTY, schedule: logistic}"
            # or simply
            -o ...maze_selection_criteria.mode=RANDOM

        Attributes:
            RANDOM (str): Choose a random maze.
            DIFFICULTY (str): Choose a maze based on difficulty. A maze is chosen
                based on the passed `schedule` method. Current support methods are
                `logistic`, `linear`, `exponential`. See the
                `MjCambrianEnv._choose_maze` for more details.
            CURRICULUM (str): Choose a maze based on a curriculum. This is similar to
                DIFFICULTY, but CURRICULUM will schedule the maze changes based on the
                current reward. As the reward nears
                `maze_selection_criteria["factor"] * max_episode_steps`, the maze
                selection will lean towards more difficult mazes.
            NAMED (str): Choose a maze based on name. `name` must be passed as a kwarg
                to the selection method.
            CYCLE (str): Cycle through the mazes. The mazes are cycled through in
                the order they are defined in the config.
        """

        RANDOM: str = "random"
        DIFFICULTY: str = "difficulty"
        CURRICULUM: str = "curriculum"
        NAMED: str = "named"
        CYCLE: str = "cycle"

    maze_selection_criteria: Dict[str, Any]
    mazes: List[MjCambrianMazeConfig]

    animals: List[MjCambrianAnimalConfig]


@config_wrapper
class MjCambrianPopulationConfig(MjCambrianBaseConfig):
    """Config for a population. Used for type hinting.

    Attributes:
        size (int): The population size. This represents the number of agents that
            should be trained at any one time.
        num_top_performers (int): The number of top performers to use in the new agent
            selection. Either in cross over or in mutation, these top performers are
            used to generate new agents.
    """

    size: int
    num_top_performers: int


@config_wrapper
class MjCambrianSpawningConfig(MjCambrianBaseConfig):
    """Config for spawning. Used for type hinting.

    Attributes:
        init_num_mutations (int): The number of mutations to perform on the
            default config to generate the initial population. The actual number of
            mutations is calculated using random.randint(1, init_num_mutations).
        num_mutations (int): The number of mutations to perform on the parent
            generation to generate the new generation. The actual number of mutations
            is calculated using random.randint(1, num_mutations).
        mutations (List[str]): The mutation options to use for the animal. See
            `MjCambrianAnimal.MutationType` for options.
        mutation_options (Optional[Dict[str, Any]]): The options to use for
            the mutations.

        load_policy (bool): Whether to load a policy or not. If True, the parent's
            saved policy will be loaded and used as the starting point for the new
            generation. If False, the child will be trained from scratch.

        replication_type (str): The type of replication to use. See
            `ReplicationType` for options.
    """

    init_num_mutations: int
    num_mutations: int
    mutations: List[str]
    mutation_options: Optional[Dict[str, Any]] = None

    load_policy: bool

    class ReplicationType(Flag):
        """Use as bitmask to specify which type of replication to perform on the animal.

        Example:
        >>> # Only mutation
        >>> type = ReplicationType.MUTATION
        >>> # Both mutation and crossover
        >>> type = ReplicationType.MUTATION | ReplicationType.CROSSOVER
        """

        MUTATION = auto()
        CROSSOVER = auto()

    replication_type: str


@config_wrapper
class MjCambrianEvoConfig(MjCambrianBaseConfig):
    """Config for evolutions. Used for type hinting.

    Attributes:
        max_n_envs (int): The maximum number of environments to use for
            parallel training. Will set `n_envs` for each training process to
            `max_n_envs // population size`.
        num_generations (int): The number of generations to run for.
        num_nodes (int): The number of nodes used for the evolution process. By default,
            this should be 1. And then if multiple evolutions are run in parallel, the
            number of nodes should be set to the number of evolutions.

        population_config (MjCambrianPopulationConfig): The config for the population.
        spawning_config (MjCambrianSpawningConfig): The config for the spawning process.

        generation_config (Optional[MjCambrianGenerationConfig]): The config for the
            current generation. Will be set by the evolution runner.
        parent_generation_config (Optional[MjCambrianGenerationConfig]): The config for
            the parent generation. Will be set by the evolution runner. If None, that
            means that the current generation is the first generation (i.e. no parent).

        top_performers (Optional[List[str]]): The top performers from the parent
            generation. This was used to select an animal to spawn an offspring from.
            Used for parsing after the fact.

        environment_variables (Optional[Dict[str, str]]): The environment variables to
            set for the training process.
    """

    max_n_envs: int
    num_generations: int
    num_nodes: int

    population_config: MjCambrianPopulationConfig
    spawning_config: MjCambrianSpawningConfig

    generation_config: Optional[MjCambrianGenerationConfig] = None
    parent_generation_config: Optional[MjCambrianGenerationConfig] = None

    top_performers: Optional[List[str]] = None

    environment_variables: Dict[str, str]


@config_wrapper
class MjCambrianConfig(MjCambrianBaseConfig):
    """The base config for the mujoco cambrian environment. Used for type hinting.

    Attributes:
        logdir (str): The directory to log training data to.
        expname (str): The name of the experiment. Used to name the logging
            subdirectory. If unset, will set to the name of the config file.

        seed (int): The base seed used when intializing the default thread/process.
            Launched processes should use this seed value to calculate their own seed
            values. This is used to ensure that each process has a unique seed.

        training_config (MjCambrianTrainingConfig): The config for the training process.
        env_config (MjCambrianEnvConfig): The config for the environment.
        evo_config (Optional[MjCambrianEvoConfig]): The config for the evolution
            process. If None, the environment will not be run in evolution mode.
        logging_config (Optional[Dict[str, Any]]): The config for the logging process.
            Passed to `logging.config.dictConfig`.
    """

    logdir: str
    expname: str

    seed: int

    training_config: MjCambrianTrainingConfig
    env_config: MjCambrianEnvConfig
    evo_config: Optional[MjCambrianEvoConfig] = None
    logging_config: Optional[Dict[str, Any]] = None


def setup_hydra(main_fn: Optional[Callable[["MjCambrianConfig"], None]] = None, /):
    """This function is the main entry point for the hydra application.

    Args:
        main_fn (Callable[["MjCambrianConfig"], None]): The main function to be called
            after the hydra configuration is parsed.
    """
    import hydra

    zen.store.add_to_hydra_store()

    def hydra_argparse_override(fn: Callable, /):
        """This function allows us to add custom argparse parameters prior to hydra
        parsing the config.

        We want to set some defaults for the hydra config here. This is a workaround
        in a way such that we don't

        Note:
            Augmented from hydra discussion #2598.
        """
        import sys
        import argparse

        parser = argparse.ArgumentParser()
        parsed_args, unparsed_args = parser.parse_known_args()

        # By default, argparse uses sys.argv[1:] to search for arguments, so update
        # sys.argv[1:] with the unparsed arguments for hydra to parse (which uses
        # argparse).
        sys.argv[1:] = unparsed_args

        return fn if fn is not None else lambda fn: fn

    @hydra_argparse_override
    @hydra.main(
        version_base=None, config_path=f"{os.getcwd()}/configs", config_name="base"
    )
    def main(cfg: DictConfig):
        OmegaConf.resolve(cfg)
        cfg.custom = None

        main_fn(zen.instantiate(cfg))

    main()


if __name__ == "__main__":
    import time

    t0 = time.time()

    def main(config: MjCambrianConfig):
        # print(config)
        config.save("config.yaml")

    setup_hydra(main)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
