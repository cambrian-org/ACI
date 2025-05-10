"""Defines the MjCambrianMazeEnv class."""

from enum import Enum
from typing import (
    Any,
    Callable,
    Concatenate,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    TypeAlias,
)

import numpy as np
from hydra_config import HydraContainerConfig, config_wrapper

from cambrian.agents.agent import MjCambrianAgent
from cambrian.envs.env import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.utils import safe_index
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.spec import MjCambrianSpec

DEFAULT_ENTITY_ID: str = "default"


class MjCambrianMapEntity(Enum):
    """
    Enum representing different states in a grid.

    Attributes:
        RESET (str): Initial reset position of the agent. Can include agent IDs in
            the format "R:<agent id>".
        WALL (str): Represents a wall in the grid. Can include texture IDs in the
            format "1:<texture id>".
        EMPTY (str): Represents an empty space in the grid.
    """

    RESET = "R"
    WALL = "1"
    EMPTY = "0"

    @staticmethod
    def parse(value: str) -> Tuple[Self, str]:
        """
        Parse a value to handle special formats like "1:<texture id>".

        Args:
            value (str): The value to parse.

        Returns:
            Tuple[Self, str]: The parsed entity and the texture id if
                applicable.
        """
        if value.startswith("1:"):
            return MjCambrianMapEntity.WALL, value[2:]
        elif value.startswith("R:"):
            return MjCambrianMapEntity.RESET, value[2:]
        for entity in MjCambrianMapEntity:
            if value == entity.value:
                return entity, DEFAULT_ENTITY_ID
        raise ValueError(f"Unknown MjCambrianMapEntity: {value}")


@config_wrapper
class MjCambrianMazeConfig(HydraContainerConfig):
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
        hflip (bool): Whether to flip the maze or not. If True, the maze will be
            flipped along the x-axis.
        vflip (bool): Whether to flip the maze or not. If True, the maze will be
            flipped along the y-axis.
        rotation (float): The rotation of the maze in degrees. The rotation is
            applied after the flip.

        wall_texture_map (Dict[str, List[str]]): The mapping from texture id to
            texture names. Textures in the list are chosen at random. If the list is of
            length 1, only one texture will be used. A length >= 1 is required.
            The keyword "default" is required for walls denoted simply as 1 or W.
            Other walls are specified as 1/W:<texture id>.
        agent_id_map (Dict[str, List[str]]): The mapping from agent id to agent
            names. Agents in the list are chosen at random. If the list is of length 1,
            only one agent will be used. A length >= 1 is required for each agent name.
            Effectively, this means you can set a reset position as R:<agent id> in the
            map and this map is used to assign to a group of agents. For instance,
            R:O in the map and including O: [agent1, agent2] in the agent_id_map will
            assign the reset position to either agent1 or agent2 at random. "default"
            is required for agents denoted simply as R.

        enabled (bool): Whether the maze is enabled or not.
    """

    xml: MjCambrianXML
    map: str

    scale: float
    height: float
    hflip: bool
    vflip: bool
    rotation: float

    wall_texture_map: Dict[str, List[str]]
    agent_id_map: Dict[str, List[str]]

    enabled: bool


MjCambrianMazeSelectionFn: TypeAlias = Callable[
    Concatenate[MjCambrianAgent, Dict[str, Any], ...], float
]


@config_wrapper
class MjCambrianMazeEnvConfig(MjCambrianEnvConfig):
    """
    mazes (Dict[str, MjCambrianMazeEnvConfig]): The configs for the mazes. Each
        maze will be loaded into the scene and the agent will be placed in a maze
        at each reset.
    maze_selection_fn (MjCambrianMazeSelectionFn): The function to use to select
        the maze. The function will be called at each reset to select the maze
        to use. See `MjCambrianMazeSelectionFn` and `maze.py` for more info.
    """

    mazes: Dict[str, MjCambrianMazeConfig]
    maze_selection_fn: MjCambrianMazeSelectionFn


class MjCambrianMazeEnv(MjCambrianEnv):
    def __init__(self, config: MjCambrianMazeEnvConfig, **kwargs):
        self._config = config

        # Have to initialize the mazes first since generate_xml is called from the
        # MjCambrianEnv constructor
        self._maze: MjCambrianMaze = None
        self._maze_store = MjCambrianMazeStore(config.mazes, config.maze_selection_fn)
        self._agent_models: []
        self._training_agent: str = None

        super().__init__(config, **kwargs)

    def set_agent_models(self, agent_models):
        self._agent_models = agent_models

    def set_training_agent(self, agent_name: str):
        """Sets which agent is currently being trained.
        
        Args:
            agent_name (str): Name of the agent to train.
        """
        self._training_agent = agent_name

    @property
    def agent_models(self):
        return self._agent_models

    @property
    def training_agent(self) -> str:
        """Returns the name of the agent currently being trained."""
        return self._training_agent

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.make_empty()

        # Add the mazes to the xml
        # Do this first so overrides defined in the env xml are applied
        xml += self._maze_store.generate_xml()

        # Add the rest of the xml
        xml += super().generate_xml()

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        # Set the random seed first
        if seed is not None:
            self.set_random_seed(seed)

        # Choose the maze
        self._maze = self._maze_store.select_maze(self)
        self._maze_store.reset(self.spec)

        # For each agent, generate an initial position
        for agent in self.agents.values():
            agent.init_pos = self._maze.generate_reset_pos(agent=agent.name)

        # Now reset the environment
        obs, info = super().reset(seed=seed, options=options)

        if (renderer := self.renderer) and (viewer := renderer.viewer):
            if viewer.config.config.get("camera", {}).get("lookat", None) is None:
                # Update the camera positioning to match the current maze
                viewer.camera.lookat = self._maze.lookat

            # Update the camera distance to match the current maze's extent
            if viewer.config.config.get("camera", {}).get("distance", None) is None:
                viewer.camera.distance = viewer.config.camera.distance
            viewer.camera.distance *= self._maze.max_dim / renderer.ratio
            if self._maze.ratio < 2.0:
                viewer.camera.distance *= 2

        return obs, info

    # ==================

    @property
    def maze(self) -> "MjCambrianMaze":
        """Returns the current maze."""
        return self._maze

    @maze.setter
    def maze(self, maze: "MjCambrianMaze"):
        """Sets the current maze."""
        self._maze = maze

    @property
    def maze_store(self) -> "MjCambrianMazeStore":
        """Returns the maze store."""
        return self._maze_store


# ================


class MjCambrianMaze:
    """The maze class. Generates a maze from a given map and provides utility
    functions for working with the maze."""

    def __init__(self, config: MjCambrianMazeConfig, name: str):
        self._config = config
        self._name = name
        self._starting_x = None

        self._map: np.ndarray = None
        self._load_map()

        self._wall_textures: List[str] = []
        self._wall_locations: List[np.ndarray] = []
        self._reset_locations: List[np.ndarray] = []
        self._reset_agents: List[str] = []
        self._occupied_locations: List[np.ndarray] = []
        self._agent_locations: Dict[str, int] = {}

        self._agent_id_map = dict(**self._config.agent_id_map)

    def initialize(self, starting_x: float):
        self._starting_x = starting_x
        self._update_locations()

    def _load_map(self):
        """Parses the map (which is a str) as a yaml str and converts it to an
        np array."""
        import yaml

        self._map = np.array(yaml.safe_load(self._config.map), dtype=str)
        if self._config.hflip:
            self._map = np.flip(self._map, axis=0)
        if self._config.vflip:
            self._map = np.flip(self._map, axis=1)
        self._map = np.rot90(self._map, k=int(self._config.rotation / 90))

    def _update_locations(self):
        """This helper method will update the initially place the wall and reset
        locations. These are known at construction time. It will also parse wall
        textures."""

        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                struct = self._map[i][j]

                # Calculate the cell location in global coords
                x = (j + 0.5) * self._config.scale - self.x_map_center
                y = self.y_map_center - (i + 0.5) * self._config.scale
                loc = np.array([x, y])

                entity, entity_id = MjCambrianMapEntity.parse(struct)
                if entity == MjCambrianMapEntity.WALL:
                    self._wall_locations.append(loc)

                    # Do a check for the texture
                    assert entity_id in self._config.wall_texture_map, (
                        f"Invalid texture: {entity_id}. "
                        f"Available textures: {list(self._config.wall_texture_map.keys())}"  # noqa
                    )
                    self._wall_textures.append(entity_id)
                elif entity == MjCambrianMapEntity.RESET:
                    self._reset_locations.append(loc)

                    # Do a check for the agent id
                    assert entity_id in list(self._agent_id_map.keys()), (
                        f"Invalid agent id: {entity_id}. "
                        f"Available agent ids: {list(self._agent_id_map.keys())}"
                    )
                    self._reset_agents.append(entity_id)

    def generate_xml(self) -> MjCambrianXML:
        xml = MjCambrianXML.from_string(self._config.xml)

        worldbody = xml.find(".//worldbody")
        assert worldbody is not None, "xml must have a worldbody tag"
        assets = xml.find(".//asset")
        assert assets is not None, "xml must have an asset tag"

        # Add the wall textures
        for t, textures in self._config.wall_texture_map.items():
            for texture in textures:
                name_prefix = f"wall_{self._name}_{t}_{texture}"
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

        # Add all walls as a single geom using a mesh
        scale = self._config.scale / 2
        height = self._config.height

        # Create a mesh for all walls
        wall_vertices = []
        wall_faces = []
        vertex_count = 0

        for i, (x, y) in enumerate(self._wall_locations):
            # Create vertices for a box
            vertices = [
                [x - scale, y - scale, 0],
                [x + scale, y - scale, 0],
                [x + scale, y + scale, 0],
                [x - scale, y + scale, 0],
                [x - scale, y - scale, height * scale],
                [x + scale, y - scale, height * scale],
                [x + scale, y + scale, height * scale],
                [x - scale, y + scale, height * scale],
            ]
            
            # Add vertices
            wall_vertices.extend(vertices)
            
            # Add faces (each face is a triangle)
            # Bottom face
            wall_faces.extend([
                [vertex_count, vertex_count + 1, vertex_count + 2],
                [vertex_count, vertex_count + 2, vertex_count + 3]
            ])
            # Top face
            wall_faces.extend([
                [vertex_count + 4, vertex_count + 5, vertex_count + 6],
                [vertex_count + 4, vertex_count + 6, vertex_count + 7]
            ])
            # Side faces
            wall_faces.extend([
                [vertex_count, vertex_count + 1, vertex_count + 5],
                [vertex_count, vertex_count + 5, vertex_count + 4],
                [vertex_count + 1, vertex_count + 2, vertex_count + 6],
                [vertex_count + 1, vertex_count + 6, vertex_count + 5],
                [vertex_count + 2, vertex_count + 3, vertex_count + 7],
                [vertex_count + 2, vertex_count + 7, vertex_count + 6],
                [vertex_count + 3, vertex_count, vertex_count + 4],
                [vertex_count + 3, vertex_count + 4, vertex_count + 7]
            ])
            
            vertex_count += 8

        # Add the mesh to assets
        mesh_name = f"wall_{self._name}_mesh"
        xml.add(
            assets,
            "mesh",
            name=mesh_name,
            vertex=" ".join(f"{v[0]} {v[1]} {v[2]}" for v in wall_vertices),
            face=" ".join(f"{f[0]} {f[1]} {f[2]}" for f in wall_faces)
        )

        # Add a single geom using the mesh
        xml.add(
            worldbody,
            "geom",
            name=f"wall_{self._name}",
            type="mesh",
            mesh=mesh_name,
            contype="1",
            conaffinity="2",
            **{"class": f"maze_wall_{self._name}"}
        )

        # Update floor size based on the map extent
        # Only done if the size is explicitly set to 0 0 0
        floor_name = f"floor_{self._name}"
        floor = xml.find(f".//geom[@name='{floor_name}']")
        assert floor is not None, f"`{floor_name}` not found"
        if floor.attrib.get("size", "0 0 0"):
            size = f"{self.map_width_scaled // 2} {self.map_length_scaled // 2} 0.1"
            floor.attrib["size"] = size
        floor.attrib["pos"] = " ".join(map(str, [-self._starting_x, 0, -0.05]))

        return xml

    def reset(self, spec: MjCambrianSpec, *, reset_occupied: bool = True):
        """Resets the maze. Will reset the wall textures and reset the occupied
        locations, if desired."""
        if reset_occupied:
            self._occupied_locations.clear()
            self._agent_locations.clear()

        self._reset_wall_textures(spec)

    def _reset_wall_textures(self, spec: MjCambrianSpec):
        """Helper method to reset the wall textures.

        All walls will have the same texture since they're now a single geom.
        """
        # Generate a random texture for all walls
        texture_id = self._wall_textures[0] if self._wall_textures else "default"
        texture_name = np.random.choice(list(self._config.wall_texture_map[texture_id]))
        
        # Update the wall geom material
        wall_name = f"wall_{self._name}"
        geom_id = spec.get_geom_id(wall_name)
        assert geom_id != -1, f"`{wall_name}` geom not found"

        # Update the geom material
        material_name = f"wall_{self._name}_{texture_id}_{texture_name}_mat"
        spec.geoms[geom_id].material = material_name

    # ==================

    def rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        x = (rowcol_pos[1] + 0.5) * self._config.scale - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self._config.scale

        return np.array([x, y])

    def xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        i = np.floor((self.y_map_center - xy_pos[1]) / self._config.scale)
        j = np.floor((xy_pos[0] + self.x_map_center) / self._config.scale)
        return np.array([i, j], dtype=int)

    def compute_optimal_path(
        self,
        start: np.ndarray,
        target: np.ndarray,
        *,
        obstacles: List[Tuple[int, int]] = [],
    ) -> np.ndarray:
        """Computes the optimal path from the start position to the target.

        Uses a BFS to find the shortest path.

        Keyword Args:
            obstacles (List[Tuple[int, int]]): The obstacles in the maze. Each
                obstacle is a tuple of (row, col). Defaults to []. Avoids these
                positions when computing the path.
        """
        from typing import Deque

        start = self.xy_to_rowcol(start)
        target = self.xy_to_rowcol(target)

        rows = self._map.shape[0]
        cols = self._map.shape[1]
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        visited[start[0]][start[1]] = True
        queue = Deque([([start], 0)])  # (path, distance)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            path, dist = queue.popleft()
            current = path[-1]
            if np.all(current == target):
                # Convert path from indices to positions
                path = [self.rowcol_to_xy(pos) for pos in path]
                path.append(self.rowcol_to_xy(target))
                return np.array(path)

            # Check all moves (left, right, up, down, and all diagonals)
            for dr, dc in moves:
                r, c = current[0] + dr, current[1] + dc
                map_entity = MjCambrianMapEntity.parse(self._map[r][c])[0]
                if (
                    0 <= r < rows
                    and 0 <= c < cols
                    and not visited[r][c]
                    and map_entity != MjCambrianMapEntity.WALL
                ):
                    # If the movement is diagonal, check that the adjacent cells are
                    # free as well so the path doesn't clip through walls
                    pr, pc = current[0], current[1]
                    if (dr, dc) in moves[4:]:
                        pc_map_entity = MjCambrianMapEntity.parse(self._map[r][pc])[0]
                        pr_map_entity = MjCambrianMapEntity.parse(self._map[pr][c])[0]
                        if (
                            pc_map_entity == MjCambrianMapEntity.WALL
                            or pr_map_entity == MjCambrianMapEntity.WALL
                        ):
                            continue

                    # Check if the cell is an obstacle
                    if (r, c) in obstacles:
                        continue

                    visited[r][c] = True
                    queue.append((path + [(r, c)], dist + 1))

        raise ValueError("No path found")

    # ==================

    def _generate_pos(self, locations: List[np.ndarray], tries: int = 20) -> np.ndarray:
        """Helper method to generate a position. The generated position must be at a
        unique location from self._occupied_locations.

        Args:
            locations (List[np.ndarray]): The locations to choose from.
            tries (int): The number of tries to attempt to find a unique position.
                Defaults to 20.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        assert len(locations) > 0, "Not enough locations to choose from"

        for _ in range(tries):
            idx = np.random.randint(low=0, high=len(locations))
            pos = locations[idx].copy()

            # Check if the position is already occupied
            for occupied in self._occupied_locations:
                if np.linalg.norm(pos - occupied) <= 0.5 * self._config.scale:
                    break
            else:
                return pos
        raise ValueError(
            f"Could not generate a unique position. {tries} tries failed. "
            f"Occupied locations: {self._occupied_locations}. "
            f"Available locations: {locations}."
        )

    def generate_reset_pos(
        self, agent: str, *, add_as_occupied: bool = True
    ) -> np.ndarray:
        """Generates a random reset position for an agent.

        Keyword Args:
            add_as_occupied (bool): Whether to add the chosen location to the
                occupied locations. Defaults to True.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        reset_locations = []
        for reset_agent, reset_pos in zip(self._reset_agents, self._reset_locations):
            if agent in self._agent_id_map[reset_agent]:
                reset_locations.append(reset_pos)
        reset_locations = np.array(reset_locations)
        if len(reset_locations) == 0:
            raise ValueError(f"No reset locations found for agent '{agent}'.")

        # Reset the occupied location if the agent is already in the occupied locations
        if agent in self._agent_locations:
            del self._occupied_locations[self._agent_locations[agent]]

        # Generate the pos and assign that pos to the agent
        reset_locations = [tuple(loc) for loc in reset_locations]
        occupied_locations = [tuple(loc) for loc in self._occupied_locations]
        possible_locations = list(set(reset_locations) - set(occupied_locations))
        pos = possible_locations[np.random.choice(len(possible_locations))]
        if add_as_occupied:
            self._agent_locations[agent] = len(self._occupied_locations)
            self._occupied_locations.append(pos)
        return pos

    # ==================

    @property
    def config(self) -> MjCambrianMazeEnvConfig:
        """Returns the config."""
        return self._config

    @property
    def name(self) -> str:
        """Returns the name."""
        return self._name

    @property
    def map(self) -> np.ndarray:
        """Returns the map."""
        return self._map

    @property
    def map_length_scaled(self) -> float:
        """Returns the map length scaled."""
        return self._map.shape[0] * self._config.scale

    @property
    def map_width_scaled(self) -> float:
        """Returns the map width scaled."""
        return self._map.shape[1] * self._config.scale

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
        assert self._starting_x is not None, "Maze has not been initialized"
        return self.map_width_scaled // 2 + self._starting_x

    @property
    def y_map_center(self) -> float:
        """Returns the y map center."""
        return self.map_length_scaled / 2

    @property
    def lookat(self) -> np.ndarray:
        """Returns a point which aids in placement of a camera to visualize this
        maze."""
        # NOTE: Negative because of convention based on BEV camera
        assert self._starting_x is not None, "Maze has not been initialized"
        return np.array([-self._starting_x + len(self._map[0]) / 4, 0, 0])

    @property
    def reset_locations(self) -> List[np.ndarray]:
        """Returns the reset locations."""
        return self._reset_locations


# ================================


class MjCambrianMazeStore:
    """This is a simple class to store a collection of mazes."""

    def __init__(
        self,
        maze_configs: Dict[str, MjCambrianMazeEnvConfig],
        maze_selection_fn: MjCambrianMazeSelectionFn,
    ):
        self._mazes: Dict[str, MjCambrianMaze] = {}
        self._create_mazes(maze_configs)

        self._current_maze: MjCambrianMaze = None
        self._maze_selection_fn = maze_selection_fn

    def _create_mazes(self, maze_configs: Dict[str, MjCambrianMazeEnvConfig]):
        prev_x, prev_width = 0, 0
        for name, config in maze_configs.items():
            if name in self._mazes:
                # If the maze already exists, skip it
                continue
            elif not config.enabled:
                # If the maze is not enabled, skip it
                continue

            # First create the maze
            maze = MjCambrianMaze(config, name)
            self._mazes[name] = maze

            # Calculate the starting x of the maze
            # We'll place the maze such that it doesn't overlap with existing mazes
            # It'll be placed next to the previous one
            # The positions of the maze is calculated from one corner (defined as x
            # in this case)
            x = prev_x + prev_width / 2 + maze.map_width_scaled / 2
            maze.initialize(x)

            # Update the prev_center and prev_width
            prev_x, prev_width = x, maze.map_width_scaled

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the current maze."""
        xml = MjCambrianXML.make_empty()

        for maze in self._mazes.values():
            xml += maze.generate_xml()

        return xml

    def reset(self, spec: MjCambrianSpec):
        """Resets all mazes."""
        for maze in self._mazes.values():
            maze.reset(spec)

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
        idx = safe_index(self.maze_list, self._current_maze, default=-1)
        return self.maze_list[(idx + 1) % len(self.maze_list)]
