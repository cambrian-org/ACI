from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    List,
    TypeAlias,
    Callable,
    Concatenate,
)
from enum import Enum

import mujoco as mj
import numpy as np

from cambrian.envs.env import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv, MjCambrianObjectEnvConfig
from cambrian.animals.animal import MjCambrianAnimal
from cambrian.utils import get_geom_id
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML


class MjCambrianMapEntity(Enum):
    """
    Enum representing different states in a grid.

    Attributes:
        RESET (str): Initial reset position of the agent.
        OBJECT (str): Possible object locations.
        WALL (str): Represents a wall in the grid. Can include texture IDs in the
            format "1:<texture id>".
        EMPTY (str): Represents an empty space in the grid.
    """

    RESET = "R"
    OBJECT = "X"
    WALL = "1"
    EMPTY = "0"

    @staticmethod
    def parse(value: str) -> Tuple[Enum, str | None]:
        """
        Parse a value to handle special formats like "1:<texture id>".

        Args:
            value (str): The value to parse.

        Returns:
            Tuple[Enum, Optional[str]]: The parsed entity and the texture id if
                applicable.
        """
        if value.startswith("1:"):
            return MjCambrianMapEntity.WALL, value[2:]
        for entity in MjCambrianMapEntity:
            if value == entity.value:
                return entity, None
        raise ValueError(f"Unknown MjCambrianMapEntity: {value}")


@config_wrapper
class MjCambrianMazeConfig(MjCambrianBaseConfig):
    """Defines a map config. Used for type hinting.

    Attributes:
        xml (MjCambrianXML): The xml for the maze. This is the xml that will be
            used to create the maze.
        map (str): The map to use for the maze. It's a 2D array where
            each element is a string and corresponds to a "pixel" in the map. See
            `maze.py` for info on what different strings mean. This is actually a
            List[List[str]], but we keep it as a string for readability when dumping
            the config to a file. Will convert to list when creating the maze.

        scale (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.
        flip (bool): Whether to flip the maze or not. If True, the maze will be
            flipped along the x-axis.
        smooth_walls (bool): Whether to smooth the walls such that they are continuous
            appearing. This is an approximated as a spline fit to the walls.

        wall_texture_map (Dict[str, List[str]]): The mapping from texture id to
            texture names. Textures in the list are chosen at random. If the list is of
            length 1, only one texture will be used. A length >= 1 is required.
            The keyword "default" is required for walls denoted simply as 1 or W.
            Other walls are specified as 1/W:<texture id>.
    """

    xml: MjCambrianXML
    map: str

    scale: float
    height: float
    flip: bool
    smooth_walls: bool

    wall_texture_map: Dict[str, List[str]]


MjCambrianMazeSelectionFn: TypeAlias = Callable[
    Concatenate[MjCambrianAnimal, Dict[str, Any], ...], float
]


@config_wrapper
class MjCambrianMazeEnvConfig(MjCambrianObjectEnvConfig):
    """
    mazes (Dict[str, MjCambrianMazeConfig]): The configs for the mazes. Each
        maze will be loaded into the scene and the animal will be placed in a maze
        at each reset.
    maze_selection_fn (MjCambrianMazeSelectionFn): The function to use to select
        the maze. The function will be called at each reset to select the maze
        to use. See `MjCambrianMazeSelectionFn` and `maze.py` for more info.
    """

    mazes: Dict[str, MjCambrianMazeConfig]
    maze_selection_fn: MjCambrianMazeSelectionFn


class MjCambrianMazeEnv(MjCambrianObjectEnv):
    def __init__(self, config: MjCambrianMazeEnvConfig):
        self.config: MjCambrianMazeEnvConfig = config

        # Have to initialize the mazes first since generate_xml is called from the
        # MjCambrianEnv constructor
        self.maze: MjCambrianMaze = None
        self.maze_store = MjCambrianMazeStore(
            self.config.mazes, self.config.maze_selection_fn
        )

        super().__init__(config)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = super().generate_xml()

        # Add the mazes to the xml
        xml += self.maze_store.generate_xml()

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        # Choose the maze
        self.maze = self.maze_store.select_maze(self)
        self.maze_store.reset(self.model)

        # For each animal, generate an initial position
        # TODO: is there a better way to do this?
        for animal in self.animals.values():
            reset_pos = self.maze.generate_reset_pos()
            for i in range(len(reset_pos)):
                animal.config.initial_qpos[i] = float(reset_pos[i])

        # For each object, generate an initial position
        for obj in self.objects.values():
            obj.config.pos[:2] = self.maze.generate_object_pos()
            obj.config.pos[2] = self.maze.config.scale // 4

        # Now reset the environment
        obs, info = super().reset(seed=seed, options=options)

        if (renderer := self.renderer) and (viewer := renderer.viewer):
            # Update the camera positioning to match the current maze
            # Only update if the camera lookat is not set in the config file
            if viewer.config.select("camera.lookat") is None:
                viewer.config.camera.lookat = self.maze.lookat

            # Update the camera distance to match the current maze's extent
            viewer.camera.distance = viewer.config.select("camera.distance", default=1)
            if self.maze.ratio < 2:
                viewer.camera.distance *= renderer.ratio * self.maze.min_dim
            else:
                viewer.camera.distance *= self.maze.max_dim / renderer.ratio

        return obs, info


# ================


class MjCambrianMaze:
    """The maze class. Generates a maze from a given map and provides utility
    functions for working with the maze."""

    def __init__(self, config: MjCambrianMazeConfig, name: str, starting_x: float):
        self.config = config
        self.name = name
        self._starting_x = starting_x

        self._map: np.ndarray = None
        self._load_map()

        self._wall_textures: List[str] = []
        self._wall_locations: List[np.ndarray] = []
        self._reset_locations: List[np.ndarray] = []
        self._object_locations: List[np.ndarray] = []
        self._occupied_locations: List[np.ndarray] = []
        self._update_locations()

    def _load_map(self):
        """Parses the map (which is a str) as a yaml str and converts it to an
        np array."""
        import yaml

        self._map = np.array(yaml.safe_load(self.config.map), dtype=str)
        if self.config.flip:
            self._map = np.flip(self._map)

    def _update_locations(self):
        """This helper method will update the initially place the wall and reset
        locations. These are known at construction time. It will also parse wall
        textures."""

        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                struct = self._map[i][j]

                # Calculate the cell location in global coords
                x = (j + 0.5) * self.config.scale - self.x_map_center
                y = self.y_map_center - (i + 0.5) * self.config.scale
                loc = np.array([x, y])

                entity, texture_id = MjCambrianMapEntity.parse(struct)
                if entity == MjCambrianMapEntity.WALL:
                    self._wall_locations.append(loc)

                    # Do a check for the texture
                    assert texture_id in self.config.wall_texture_map, (
                        f"Invalid texture: {texture_id}. "
                        f"Available textures: {self.config.wall_texture_map.keys()}"
                    )
                    self._wall_textures.append(texture_id)
                elif entity == MjCambrianMapEntity.RESET:
                    self._reset_locations.append(loc)
                elif entity == MjCambrianMapEntity.OBJECT:
                    self._object_locations.append(loc)

    def generate_xml(self) -> MjCambrianXML:
        xml = MjCambrianXML.from_config(self.config.xml)

        worldbody = xml.find(".//worldbody")
        assert worldbody is not None, "xml must have a worldbody tag"
        assets = xml.find(".//asset")
        assert assets is not None, "xml must have an asset tag"

        # Add the wall textures
        for t, textures in self.config.wall_texture_map.items():
            for texture in textures:
                name_prefix = f"wall_{self.name}_{t}_{texture}"
                xml.add(
                    assets,
                    "material",
                    name=f"{name_prefix}_mat",
                    texture=f"{name_prefix}_tex",
                )
                xml.add(
                    assets,
                    "texture",
                    name=f"{name_prefix}_tex",
                    file=f"maze_textures/{texture}.png",
                    gridsize="3 4",
                    gridlayout=".U..LFRB.D..",
                )

        # Add the walls. Each wall has it's own geom.
        scale = self.config.scale / 2
        height = self.config.height
        for i, (x, y) in enumerate(self._wall_locations):
            name = f"wall_{self.name}_{i}"
            xml.add(
                worldbody,
                "geom",
                name=name,
                pos=f"{x} {y} {scale * height}",
                size=f"{scale} {scale} {scale * height}",
                **{"class": f"maze_wall_{self.name}"},
            )

        # Update floor size based on the map extent
        floor_name = f"floor_{self.name}"
        floor = xml.find(f".//geom[@name='{floor_name}']")
        assert floor is not None, f"`{floor_name}` not found"
        floor.attrib[
            "size"
        ] = f"{self.map_width_scaled // 2} {self.map_length_scaled // 2} 0.1"
        floor.attrib["pos"] = " ".join(map(str, [*self.lookat[:2], -0.05]))

        return xml

    def reset(self, model: mj.MjModel, *, reset_occupied: bool = True):
        """Resets the maze. Will reset the wall textures and reset the occupied
        locations, if desired."""
        if reset_occupied:
            self._occupied_locations.clear()

        self._reset_wall_textures(model)

    def _reset_wall_textures(self, model: mj.MjModel):
        """Helper method to reset the wall textures.

        All like-labelled walls will have the same texture. Their textures will be
        randomly selected from their respective texture lists.
        """

        # First, generate the texture_id -> texture_name mapping
        texture_map: Dict[str, str] = {}
        for t in self._wall_textures:
            if t not in texture_map:
                texture_map[t] = np.random.choice(list(self.config.wall_texture_map[t]))

        # Now, update the wall textures
        for i, t in zip(range(len(self._wall_locations)), self._wall_textures):
            wall_name = f"wall_{self.name}_{i}"
            geom_id = get_geom_id(model, wall_name)
            assert geom_id != -1, f"`{wall_name}` geom not found"

            # Randomly select a texture for the wall
            material_name = f"wall_{self.name}_{t}_{texture_map[t]}_mat"
            material_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, material_name)
            assert material_id != -1, f"`{material_name}` material not found"

            # Update the geom material
            model.geom_matid[geom_id] = material_id

    # ==================

    def _generate_pos(
        self,
        locations: List[np.ndarray],
        add_as_occupied: bool = True,
        tries: int = 20,
    ) -> np.ndarray:
        """Helper method to generate a position. The generated position must be at a
        unique location from self._occupied_locations.

        Args:
            locations (List[np.ndarray]): The locations to choose from.
            add_as_occupied (bool): Whether to add the chosen location to the
                occupied locations. Defaults to True.
            tries (int): The number of tries to attempt to find a unique position.
                Defaults to 20.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        assert len(locations) > 0, "No locations to choose from"

        for _ in range(tries):
            idx = np.random.randint(low=0, high=len(locations))
            pos = locations[idx].copy()

            # Check if the position is already occupied
            for occupied in self._occupied_locations:
                if np.linalg.norm(pos - occupied) <= 0.5 * self.config.scale:
                    break
            else:
                if add_as_occupied:
                    self._occupied_locations.append(pos)
                return pos
        raise ValueError(
            f"Could not generate a unique position. {tries} tries failed. "
            f"Occupied locations: {self._occupied_locations}. "
            f"Available locations: {locations}."
        )

    def generate_reset_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random reset position for an agent.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        return self._generate_pos(self._reset_locations, add_as_occupied)

    def generate_object_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random object position.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        return self._generate_pos(self._object_locations, add_as_occupied)

    # ==================

    @property
    def map_length_scaled(self) -> float:
        """Returns the map length scaled."""
        return self._map.shape[0] * self.config.scale

    @property
    def map_width_scaled(self) -> float:
        """Returns the map width scaled."""
        return self._map.shape[1] * self.config.scale

    @property
    def max_dim(self) -> float:
        """Returns the max dimension."""
        return max(self.map_length_scaled, self.map_width_scaled)

    @property
    def min_dim(self) -> float:
        """Returns the min dimension."""
        return min(self.map_length_scaled, self.map_width_scaled)

    @property
    def ratio(self) -> float:
        """Returns the ratio of the length over width."""
        return self.map_length_scaled / self.map_width_scaled

    @property
    def x_map_center(self) -> float:
        """Returns the x map center."""
        return self.map_width_scaled // 2 + self._starting_x

    @property
    def y_map_center(self) -> float:
        """Returns the y map center."""
        return self.map_length_scaled / 2

    @property
    def lookat(self) -> np.ndarray:
        """Returns a point which aids in placement of a camera to visualize this maze."""
        # TODO: Negative because of convention based on BEV camera
        return np.array([-self._starting_x, 0, 0])


# ================================


class MjCambrianMazeStore:
    """This is a simple class to store a collection of mazes."""

    def __init__(
        self,
        maze_configs: Dict[str, MjCambrianMazeConfig],
        maze_selection_fn: MjCambrianMazeSelectionFn,
    ):
        self._mazes: Dict[str, MjCambrianMaze] = {}
        self._create_mazes(maze_configs)

        self._current_maze: MjCambrianMaze = None
        self._maze_selection_fn = maze_selection_fn

    def _create_mazes(self, maze_configs: Dict[str, MjCambrianMazeConfig]):
        prev_x, prev_width = 0, 0
        for name, config in maze_configs.items():
            if name in self._mazes:
                # If the maze already exists, skip it
                continue

            # Calculate the starting x of the maze
            # We'll place the maze such that it doesn't overlap with existing mazes
            # It'll be placed next to the previous one
            # The positions of the maze is calculated from one corner (defined as x
            # in this case)
            x = prev_x + prev_width / 2 + config.scale

            # Now create the maze
            self._mazes[name] = MjCambrianMaze(config, name, x)

            # Update the prev_center and prev_width
            prev_x, prev_width = x, self._mazes[name].map_width_scaled

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the current maze."""
        xml = MjCambrianXML.make_empty()

        for maze in self._mazes.values():
            xml += maze.generate_xml()

        return xml

    def reset(self, model: mj.MjModel):
        """Resets all mazes."""
        for maze in self._mazes.values():
            maze.reset(model)

    @property
    def current_maze(self) -> MjCambrianMaze:
        """Returns the current maze."""
        return self._current_maze

    @property
    def maze_list(self) -> List[MjCambrianMaze]:
        """Returns the list of mazes."""
        return list(self._mazes.values())

    # ======================
    # Maze selection methods

    def select_maze(self, env: "MjCambrianEnv") -> MjCambrianMaze:
        """This should be called by the environment to select a maze."""
        maze = self._maze_selection_fn(self, env)
        self._current_maze = maze
        return maze

    def select_maze_random(self, _: "MjCambrianEnv") -> MjCambrianMaze:
        """Selects a maze at random."""
        return np.random.choice(self.maze_list)

    def select_maze_schedule(
        self,
        env: "MjCambrianEnv",
        *,
        schedule: Optional[str] = "linear",
        total_timesteps: int,
        n_envs: int,
        lam_0: Optional[float] = -2.0,
        lam_n: Optional[float] = 2.0,
    ) -> MjCambrianMaze:
        """Selects a maze based on a schedule. The scheduled selections are based on
        the order of the mazes in the list.

        Keyword Args:
            schedule (Optional[str]): The schedule to use. One of "linear",
                "exponential", or "logistic". Defaults to "linear".

            total_timesteps (int): The total number of timesteps in the training
                schedule. Unused if schedule is None. Required otherwise.
            n_envs (int): The number of environments. Unused if schedule is None.
                Required otherwise.
            lam_0 (Optional[float]): The lambda value at the start of the schedule.
                Unused if schedule is None.
            lam_n (Optional[float]): The lambda value at the end of the schedule.
                Unused if schedule is None.
        """

        assert lam_0 < lam_n, "lam_0 must be less than lam_n"

        # Compute the current step
        steps_per_env = total_timesteps // n_envs
        step = env.num_timesteps / steps_per_env

        # Compute the lambda value
        if schedule == "linear":
            lam = lam_0 + (lam_n - lam_0) * step
        elif schedule == "exponential":
            lam = lam_0 * (lam_n / lam_0) ** (step / n_envs)
        elif schedule == "logistic":
            lam = lam_0 + (lam_n - lam_0) / (1 + np.exp(-2 * step / n_envs))
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        p = np.exp(lam * np.arange(len(self.maze_list)))
        return np.random.choice(self.maze_list, p=p / p.sum())

    def select_maze_cycle(self, env: "MjCambrianEnv") -> MjCambrianMaze:
        """Selects a maze based on a cycle."""
        idx = self.maze_list.index(self._current_maze) if self._current_maze else -1
        return self.maze_list[(idx + 1) % len(self.maze_list)]
