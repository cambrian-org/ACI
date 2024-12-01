"""Defines agent classes."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Self, Tuple

import mujoco as mj
import numpy as np
from gymnasium import spaces

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.renderer.render_utils import add_white_border
from cambrian.utils import (
    MjCambrianActuator,
    MjCambrianGeometry,
    MjCambrianJoint,
    generate_sequence_from_range,
    get_body_id,
    get_geom_id,
)
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.config import MjCambrianBaseConfig, config_wrapper
from cambrian.utils.logger import get_logger

if TYPE_CHECKING:
    from cambrian.envs import MjCambrianEnv


@config_wrapper
class MjCambrianAgentConfig(MjCambrianBaseConfig):
    """Defines the config for an agent. Used for type hinting.

    Attributes:
        instance (Callable[[Self, str, int], "MjCambrianAgent"]): The class instance
            for the agent. This is used to create the agent. Takes the config, the
            name, and the index of the agent as arguments.

        trainable (bool): Whether the agent is trainable or not. If the agent is
            trainable, it's observations will be included in the observation space
            of the environment and the model's output actions will be applied to the
            agent. If the agent is not trainable, the agent's policy can be defined
            by overriding the `get_action_privileged` method.
        use_privileged_action (bool): This is similar to `trainable`, but the agent's
            action and observation spaces is included in the environment's action
            and observation spaces, respectively. This is useful for agents that
            are trainable, but have some special logic that needs to be implemented
            in the `get_action_privileged` method. `trainable` takes precedence over
            this attribute, as in if `trainable` is False, this attribute is ignored.
        overlay_color (Tuple[float, float, float, float]): The color to
            use in the visualization of the agent.
        overlay_size (float): The size of the overlay in the visualization of the agent.

        xml (MjCambrianXMLConfig): The xml for the agent. This is the xml that will be
            used to create the agent. You should use ${parent:xml} to generate
            named attributes. This will search upwards in the yaml file to find the
            name of the agent.

        body_name (str): The name of the body that defines the main body of the agent.
        geom_name (str): The name of the geom that are used for eye placement.
        check_contacts (bool): Whether to check contacts or not. If this is True, then
            the contacts will be checked in the environment.

        init_pos (Tuple[float | None]): The initial position of the agent. Specific
            indices of the position are set when not None. The length of the tuple
            should be <= 3. None's are filled in at the end if the length is less than
            3.
        init_quat (Tuple[float | None]): The initial quaternion of the agent. Specific
            indices of the quaternion are set when not None. The length of the tuple
            should be <= 4. None's are filled in at the end if the length is less than
            4.
        perturb_init_pose (bool): Whether to perturb the initial pose of the agent or
            not. If this is True, then the initial pose of the agent will be randomly
            adjusted based on a normal distribution.

        use_action_obs (bool): Whether to use the action observation or not. NOTE: If
            the MjCambrianConstantActionWrapper is used, this is not reflected in the
            observation, as in the actions will vary in the observation.
        use_contact_obs (bool): Whether to use the contact observation or not. If this
            is True, then the contacts will be included in the observation space of the
            agent.

        eyes_lat_range (Optional[Tuple[float, float]]): The x range of the eye. This is
            used to determine the placement of the eye on the agent. Specified in
            degrees. This is the latitudinal/vertical range of the evenly placed eye
            about the agent's bounding sphere.
        eyes_lon_range (Optional[Tuple[float, float]]): The y range of the eye. This is
            used to determine the placement of the eye on the agent. Specified in
            degrees. This is the longitudinal/horizontal range of the evenly placed eye
            about the agent's bounding sphere.
        num_eyes_to_generate (Optional[Tuple[int, int]]): The num of eyes to generate.
            If this is specified, then the eyes will be generated on a spherical
            grid. The first element is the number of eyes to generate latitudinally and
            the second element is the number of eyes to generate longitudinally. The
            eyes will be named sequentially starting from `eye_0`. Each eye will default
            to use the first eye config in the `eyes` attribute. `eyes` must have a
            length of 1 if this is specified. Each eye is named `eye_{lat}_{lon}` where
            `lat` is the latitude index and `lon` is the longitude index.
        eyes (Dict[str, MjCambrianEyeConfig]): The eyes on the agent. The keys are the
            names of the eyes and the values are the configs for the eyes. The eyes will
            be placed on the agent at the specified coordinates.
    """

    instance: Callable[[Self, str, int], "MjCambrianAgent"]

    trainable: bool
    use_privileged_action: bool
    overlay_color: Tuple[float, float, float, float]
    overlay_size: float

    xml: MjCambrianXMLConfig

    body_name: str
    geom_name: str
    check_contacts: bool

    init_pos: Tuple[float | None]
    init_quat: Tuple[float | None]
    perturb_init_pose: bool

    use_action_obs: bool
    use_contact_obs: bool

    eyes_lat_range: Optional[Tuple[float, float]] = None
    eyes_lon_range: Optional[Tuple[float, float]] = None
    num_eyes_to_generate: Optional[Tuple[int, int]] = None
    eyes: Dict[str, MjCambrianEyeConfig]


class MjCambrianAgent:
    """The agent class is defined as a physics object with eyes.

    This object serves as an agent in a multi-agent mujoco environment. Therefore,
    it must have a uniquely identifiable name.

    In our context, an agent has at any number of eyes and a body which an eye can be
    attached to. This class abstracts away the inner workings of the mujoco model itself
    to the xml generation/loading. It uses existing xml files that only define the
    agents, with which are going to be accumulating into one large xml file that will
    be loaded into mujoco.

    To support specific agent types, you should define subclasses that include agent
    specific configs (i.e. model_path, num_joints).

    Args:
        config (MjCambrianAgentConfig): The configuration for the agent.
        name (str): The name of the agent. This is used to identify the agent in the
            environment.
    """

    def __init__(self, config: MjCambrianAgentConfig, name: str):
        self._config = self._check_config(config)
        self._name = name
        self._logger = get_logger()

        self._eyes: Dict[str, MjCambrianEye] = {}

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._init_pos: Tuple[float | None, float | None, float | None] = None
        self._init_quat: Tuple[
            float | None, float | None, float | None, float | None
        ] = None
        self._actuators: List[MjCambrianActuator] = []
        self._joints: List[MjCambrianJoint] = []
        self._geom: MjCambrianGeometry = None
        self._actadrs: List[int] = []
        self._body_id: int = None
        self._initialize()

    def _check_config(self, config: MjCambrianAgentConfig) -> MjCambrianAgentConfig:
        """Run some checks/asserts on the config to make sure everything's there."""

        return config

    def _initialize(self):
        """Initialize the agent.

        This method does the following:
            - load the base xml to MjModel
            - parse the geometry
            - place eyes at the appropriate locations
        """
        model = mj.MjModel.from_xml_string(self._config.xml)

        self._parse_geometry(model)
        self._parse_actuators(model)

        self._place_eyes()

        self._init_pos = [None] * 3
        self._init_quat = [None] * 4

        # Explicitly delete the model; probably not required, but just to be safe
        del model

    def _parse_geometry(self, model: mj.MjModel):
        """Parse the geometry to get the root body, number of controls, joints, and
        actuators. We're going to do some preprocessing of the model here to get info
        regarding num joints, num controls, etc. This is because we need to know this
        to compute the observation and action spaces, which is needed _before_ actually
        initializing mujoco (but we can't get this information until after mujoco is
        initialized and we don't want to hardcode this for extensibility).

        Note:
            We can't grab the ids/adrs here because they'll be different once we load
            the entire model
        """

        # Num of controls
        if self.config.trainable:
            assert model.nu > 0, "Trainable agent must have controllable actuators."

        # Get number of qpos/qvel/ctrl
        # Just stored for later, like to get the observation space, etc.
        self._numqpos = model.nq
        self._numctrl = model.nu

        # Create the geometries we will use for eye placement
        geom_id = get_geom_id(model, self._config.geom_name)
        assert geom_id != -1, f"Could not find geom {self._config.geom_name}."
        geom_rbound = model.geom_rbound[geom_id]
        geom_pos = model.geom_pos[geom_id]

        self._geom = MjCambrianGeometry(geom_id, geom_rbound, geom_pos)

    def _parse_actuators(self, model: mj.MjModel):
        """Parse the current model/xml for the actuators.

        We have to do this twice: once on the initial model load to get the ctrl limits
        on the actuators (for the obs space). And then later to acquire the actual
        ids/adrs.
        """
        # Root body for the agent
        body_name = self._config.body_name
        body_id = get_body_id(model, body_name)
        assert body_id != -1, f"Could not find body with name {body_name}."

        # Mujoco doesn't have a neat way to grab the actuators associated with a
        # specific agent/body, so we'll try to grab them dynamically by checking the
        # transmission ids (the joint/site adrs associated with that actuator) and
        # seeing if that the corresponding root body is on this agent's body.
        self._actuators: List[MjCambrianActuator] = []
        for actadr, ((trnid, _), trntype) in enumerate(
            zip(model.actuator_trnid, model.actuator_trntype)
        ):
            # Grab the body id associated with the actuator
            if trntype == mj.mjtTrn.mjTRN_JOINT:
                act_bodyid = model.jnt_bodyid[trnid]
            elif trntype == mj.mjtTrn.mjTRN_SITE:
                act_bodyid = model.site_bodyid[trnid]
            elif trntype == mj.mjtTrn.mjTRN_BODY:
                act_bodyid = trnid
            elif trntype == mj.mjtTrn.mjTRN_TENDON:
                act_bodyid = model.tendon_adr[trnid]
            else:
                raise NotImplementedError(f'Unsupported trntype "{trntype}".')

            # Add the actuator if the rootbody of the actuator is the same body
            # as the transimission's body
            act_rootbodyid = model.body_rootid[act_bodyid]
            if act_rootbodyid == body_id:
                ctrlrange = model.actuator_ctrlrange[actadr]
                self._actuators.append(MjCambrianActuator(actadr, trnid, *ctrlrange))

        # Get the joints
        # We use the joints to get the qpos/qvel as observations (joint specific states)
        self._joints: List[MjCambrianJoint] = []
        for jntadr, jnt_bodyid in enumerate(model.jnt_bodyid):
            jnt_rootbodyid = model.body_rootid[jnt_bodyid]
            if jnt_rootbodyid == body_id:
                # This joint is associated with this agent's body
                self._joints.append(MjCambrianJoint.create(model, jntadr))

        assert (
            len(self._joints) > 0
        ), f"Body {body_name} has no joints. Joints are required for positioning."
        if self.config.trainable:
            assert len(self._actuators) > 0, f"Body {body_name} has no actuators."

    def _place_eyes(self):
        """Place the eyes on the agent."""

        eye_configs: Dict[str, MjCambrianEyeConfig] = {}
        if num_eyes := self._config.num_eyes_to_generate:
            assert len(num_eyes) == 2, "num_eyes should be a tuple of length 2."
            assert len(self._config.eyes) == 1, "Only one eye should be specified."
            assert self._config.eyes_lat_range is not None
            assert self._config.eyes_lon_range is not None

            base_eye_name, base_eye_config = list(self._config.eyes.items())[0]

            # Place the eyes uniformly on a spherical grid. The number of latitude and
            # longitudinally bins is defined by the two attributes in `eyes`,
            # respectively.
            nlat, nlon = self._config.num_eyes_to_generate
            lat_bins = generate_sequence_from_range(self._config.eyes_lat_range, nlat)
            lon_bins = generate_sequence_from_range(self._config.eyes_lon_range, nlon)
            for lat_idx, lat in enumerate(lat_bins):
                for lon_idx, lon in enumerate(lon_bins):
                    eye_name = f"{base_eye_name}_{lat_idx}_{lon_idx}"
                    eye_config = base_eye_config.copy()
                    with eye_config.set_readonly_temporarily(False):
                        eye_config.update("coord", [lat, lon])
                    eye_configs[eye_name] = eye_config
        else:
            eye_configs = self._config.eyes

        for name, eye_config in eye_configs.items():
            self._eyes[name] = eye_config.instance(eye_config, f"{self._name}_{name}")

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the agent. Will generate the xml from the model file
        and then add eyes to it.
        """
        xml = MjCambrianXML.from_string(self._config.xml)

        # Add eyes
        for eye in self.eyes.values():
            xml += eye.generate_xml(xml, self.geom, self._config.body_name)

        return xml

    def apply_action(self, actions: List[float]):
        """Applies the action to the agent. This probably happens before step
        so that the observations reflect the state of the agent after the new action
        is applied.

        It is assumed that the actions are normalized between -1 and 1.
        """
        for action, actuator in zip(actions, self._actuators):
            # Map from -1, 1 to ctrlrange
            action = np.interp(action, [-1, 1], [actuator.low, actuator.high])
            self._data.ctrl[actuator.adr] = action

    def get_action_privileged(self, env: "MjCambrianEnv") -> List[float]:
        """This is a deviation from the standard gym API. This method is similar to
        step, but it has "privileged" access to information such as the environment.
        This method can be overridden by agents which are not trainable and need to
        implement custom step logic.

        Args:
            env (MjCambrianEnv): The environment that the agent is in. This can be
                used to get information about the environment.

        Returns:
            List[float]: The action to take.
        """
        raise NotImplementedError(
            "This method should be overridden by the subclass "
            "and should never reach here."
        )

    def reset(self, model: mj.MjModel, data: mj.MjData) -> Dict[str, Any]:
        """Sets up the agent in the environment. Uses the model/data to update
        positions during the simulation.
        """
        self._model = model
        self._data = data

        # Parse actuators; this is the second time we're doing this, but we need to
        # get the adrs of the actuators in the current model
        self._parse_actuators(model)

        # Accumulate the qpos/qvel/act adrs
        self._reset_adrs(model)

        # Reset the init pose to the config's
        self.init_pos = self._config.init_pos
        self.init_quat = self._config.init_quat

        # Update the agent's qpos
        self.pos = self._init_pos
        self.quat = self._init_quat

        if self._config.perturb_init_pose:
            self.perturb_init_pose()

        # step here so that the observations are updated
        mj.mj_forward(model, data)

        self.init_pos = self.pos
        self.init_quat = self.quat

        obs: Dict[str, Any] = {}
        for name, eye in self.eyes.items():
            obs[name] = eye.reset(model, data)

        return self._update_obs(obs)

    def _reset_adrs(self, model: mj.MjModel):
        """Resets the adrs for the agent. This is used when the model is reloaded."""

        # Root body for the agent
        body_name = self._config.body_name
        self._body_id = get_body_id(model, body_name)
        assert self._body_id != -1, f"Could not find body with name {body_name}."

        # Geometry id
        geom_id = get_geom_id(model, self._config.geom_name)
        assert geom_id != -1, f"Could not find geom {self._config.geom_name}."
        self._geom.id = geom_id

        # Accumulate the qposadrs
        self._qposadrs = []
        for joint in self._joints:
            self._qposadrs.extend(joint.qposadrs)

        self._actadrs: List[int] = [act.adr for act in self._actuators]

        assert (
            len(self._qposadrs) == self._numqpos
        ), f"Mismatch in qpos adrs for agent '{self.name}': "
        f"{len(self._qposadrs)} != {self._numqpos}."
        assert (
            len(self._actadrs) == self._numctrl
        ), f"Mismatch in actuator adrs for agent '{self.name}': "
        f"{len(self._actadrs)} != {self._numctrl}."

    def step(self) -> Dict[str, Any]:
        """Steps the eyes and returns the observation."""

        obs: Dict[str, Any] = {}
        for name, eye in self.eyes.items():
            obs[name] = eye.step()

        return self._update_obs(obs)

    def _update_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional attributes to the observation."""
        if self._config.use_action_obs:
            obs["action"] = self.last_action
        if self._config.use_contact_obs:
            obs["contacts"] = self.has_contacts

        return obs

    def create_composite_image(self) -> np.ndarray | None:
        """Creates a composite image from the eyes. If there are no eyes, then this
        returns None.

        Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
            TL T TR
            ML M MR
            BL B BR

        Each eye has a white border around it.
        """
        if self.num_eyes == 0:
            return

        from cambrian.renderer import resize_with_aspect_fill

        max_res = (
            max([eye.config.resolution[1] for eye in self.eyes.values()]),
            max([eye.config.resolution[0] for eye in self.eyes.values()]),
        )

        # Sort the eyes based on their lat/lon
        images: Dict[float, Dict[float, np.ndarray]] = {}
        for eye in self.eyes.values():
            lat, lon = eye.config.coord
            if lat not in images:
                images[lat] = {}
            assert lon not in images[lat], f"Duplicate eye at {lat}, {lon}."

            # Add the image to the dictionary
            min_dim = min(eye.prev_obs.shape[:2])
            obs = eye.prev_obs[:, :, :3]
            if min_dim > 10:
                obs = add_white_border(obs, min_dim // 10)
            images[lat][lon] = obs

        # Construct the composite image
        # Loop through the sorted list of images based on lat/lon
        composite = []
        for lat in sorted(images.keys())[::-1]:
            row = []
            for lon in sorted(images[lat].keys())[::-1]:
                row.append(resize_with_aspect_fill(images[lat][lon], *max_res))
            composite.append(np.vstack(row))
        composite = np.hstack(composite)

        if composite.size == 0:
            self._logger.warning(
                f"agent `{self.name}` observations. "
                "Maybe you forgot to call `render`?."
            )
            return None

        return composite

    @property
    def has_contacts(self) -> bool:
        """Returns whether or not the agent has contacts.

        Walks through all the contacts in the environment and checks if any of them
        involve this agent.
        """
        if not self._config.check_contacts:
            return False

        for contact in self._data.contact:
            geom1 = int(contact.geom[0])
            body1 = self._model.geom_bodyid[geom1]
            rootbody1 = self._model.body_rootid[body1]

            geom2 = int(contact.geom[1])
            body2 = self._model.geom_bodyid[geom2]
            rootbody2 = self._model.body_rootid[body2]

            is_this_agent = rootbody1 == self._body_id or rootbody2 == self._body_id
            if not is_this_agent or contact.exclude:
                # Not a contact with this agent
                continue

            return True

        return False

    @property
    def observation_space(self) -> spaces.Space:
        """The observation space is defined on an agent basis. the `env` should combine
        the observation spaces such that it's supported by stable_baselines3/pettingzoo.

        The agent has three observation spaces:
            - {eye.name}: The eyes observations
            - action: The last action that was applied to the agent
            - contacts: Whether the agent has contacts or not
        """
        observation_space: Dict[Any, spaces.Space] = {}

        for name, eye in self.eyes.items():
            observation_space[name] = eye.observation_space

        if self._config.use_action_obs:
            observation_space["action"] = self.action_space
        if self._config.use_contact_obs:
            observation_space["contacts"] = spaces.Discrete(2)

        return spaces.Dict(observation_space)

    @property
    def action_space(self) -> spaces.Space:
        """The action space is simply the controllable actuators of the agent."""
        return spaces.Box(low=-1, high=1, shape=(self._numctrl,), dtype=np.float32)

    @property
    def config(self) -> MjCambrianAgentConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._name

    @property
    def eyes(self) -> Dict[str, MjCambrianEye]:
        return self._eyes

    @property
    def num_eyes(self) -> int:
        return len(self._eyes)

    @property
    def init_pos(self) -> np.ndarray:
        """Returns the initial position of the agent."""
        return self._init_pos

    @init_pos.setter
    def init_pos(self, value: Tuple[float | None, float | None, float | None] | None):
        """Sets the initial position of the agent."""
        if value is None:
            self._init_pos = [None] * 3
            return

        for idx, val in enumerate(value):
            if val is not None:
                self._init_pos[idx] = val

    @property
    def init_quat(self) -> np.ndarray:
        """Returns the initial quaternion of the agent."""
        return self._init_quat

    @init_quat.setter
    def init_quat(
        self,
        value: Tuple[float | None, float | None, float | None, float | None] | None,
    ):
        """Sets the initial quaternion of the agent."""
        if value is None:
            self._init_quat = [None] * 4
            return

        for idx, val in enumerate(value):
            if val is not None:
                self._init_quat[idx] = val

    @property
    def qpos(self) -> np.ndarray:
        """Gets the qpos of the agent. The qpos is the state of the joints defined
        in the agent's xml. This method is used to get the state of the qpos. It's
        actually a masked array where the entries are masked if the qpos adr is not
        associated with the agent. This allows the return value to be indexed
        and edited as if it were the full qpos array."""
        mask = np.ones(self._data.qpos.shape, dtype=bool)
        mask[self._qposadrs] = False
        return np.ma.masked_array(self._data.qpos, mask=mask)

    @qpos.setter
    def qpos(self, value: np.ndarray[float | None]):
        """Set's the qpos of the agent. The qpos is the state of the joints defined
        in the agent's xml. This method is used to set the state of the qpos. The
        value input is a numpy array where the entries are either values to set
        to the corresponding qpos adr or None.

        It's allowed for `value` to be less than the total number of joints in the
        agent. If this is the case, only the first `len(value)` joints will be
        updated.
        """
        for idx, val in enumerate(value):
            if val is not None:
                self._data.qpos[self._qposadrs[idx]] = val

    @property
    def pos(self) -> np.ndarray:
        """Returns the position of the agent in the environment.

        Note:
            the returned value, if edited, doesn't not directly impact the simulation.
            To set the position of the agent, use the `pos` setter.
        """
        return self._data.xpos[self._body_id].copy()

    @pos.setter
    def pos(self, value: Tuple[float | None, float | None, float | None]):
        """Sets the position of the agent in the environment. The value is a tuple
        of the x, y, and z positions. If the value is None, the position is not
        updated.

        Note:
            This base implementation assumes the first 3 values of the qpos are the
            x, y, and z positions of the agent. This may not be the case and depends on
            the joints defined in the agent, so this method should be overridden in the
            subclass if this is not the case.
        """
        for idx, val in enumerate(value):
            if val is not None:
                self._data.qpos[self._qposadrs[idx]] = val

    @property
    def quat(self) -> np.ndarray:
        """Returns the quaternion of the agent in the environment. Fmt: `wxyz`.

        Note:
            The returned value, if edited, doesn't not directly impact the simulation.
            To set the quaternion of the agent, use the `quat` setter.
        """
        return self._data.xquat[self._body_id].copy()

    @quat.setter
    def quat(
        self, value: Tuple[float | None, float | None, float | None, float | None]
    ):
        """Sets the quaternion of the agent in the environment. The value is a tuple
        of the x, y, z, and w values. If the value is None, the quaternion is not
        updated.

        Note:
            This base implementation assumes the 3,4,5,6 indices of the qpos are the
            x, y, z, and w values of the quaternion of the agent. This may not be the
            case and depends on the joints defined in the agent, so this method should
            be overridden in the subclass if this is not the case.
        """
        for idx, val in enumerate(value):
            if val is not None:
                self._data.qpos[self._qposadrs[3 + idx]] = val

    def perturb_init_pose(self):
        """Base implementation of the pose perturbation. Doesn't make any assumptions
        about the qpos structure. For fine tuned adjustment of perturbation behavior,
        this method should be overridden in the subclass."""
        mask = np.ones(self._data.qpos.shape, dtype=bool)
        mask[self._qposadrs] = False

        # Perturb at a normal distribution with a std equal to the rbound
        self._data.qpos[~mask] += np.random.normal(
            0, self.geom.rbound, len(self._qposadrs)
        )

    @property
    def mat(self) -> np.ndarray:
        """Returns the rotation matrix of the agent in the environment."""
        return self._data.xmat[self._body_id].reshape(3, 3).copy()

    @property
    def speed(self) -> float:
        """Returns the speed of the agent in the environment."""
        return np.linalg.norm(self._data.qvel[self._qposadrs])

    @property
    def last_action(self) -> np.ndarray:
        """Returns the last action that was applied to the agent."""
        last_ctrl = self._data.ctrl.copy()
        for act in self._actuators:
            last_ctrl[act.adr] = np.interp(
                last_ctrl[act.adr], [act.low, act.high], [-1, 1]
            )
        return last_ctrl[self._actadrs]

    @property
    def geom(self) -> MjCambrianGeometry:
        """Returns the geom of the agent."""
        return self._geom

    @property
    def trainable(self) -> bool:
        """Returns whether the agent is trainable or not."""
        return self._config.trainable


class MjCambrianAgent2D(MjCambrianAgent):
    """Assumes the agent is moving on 2D plane and has a yaw hinge joint which is used
    to adjust orientation of the agent."""

    @MjCambrianAgent.quat.setter
    def quat(
        self, value: Tuple[float | None, float | None, float | None, float | None]
    ):
        """Overrides the base implementation to set the z rotation."""
        assert len(value) == 4, f"Quaternion must have 4 elements, got {len(value)}."
        # Only set quat if all values are not None
        if any(val is None for val in value):
            return

        self.qpos[self._qposadrs[2]] = np.arctan2(
            2 * (value[0] * value[3] + value[1] * value[2]),
            1 - 2 * (value[2] ** 2 + value[3] ** 2),
        )
