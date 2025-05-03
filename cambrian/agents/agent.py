"""Defines agent classes."""

from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Self, Tuple

import mujoco as mj
import numpy as np
from gymnasium import spaces
from hydra_config import HydraContainerConfig, config_wrapper

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.renderer.overlays import MjCambrianViewerOverlay
from cambrian.utils import (
    MjCambrianActuator,
    MjCambrianGeometry,
    MjCambrianJoint,
    get_logger,
)
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.spec import MjCambrianSpec, spec_from_xml_string
from cambrian.utils.types import ActionType, ObsType

if TYPE_CHECKING:
    from cambrian.envs import MjCambrianEnv


@config_wrapper
class MjCambrianAgentConfig(HydraContainerConfig):
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
        perturb_init_pos (bool): Whether to perturb the initial pos of the agent or
            not. If this is True, then the initial pos of the agent will be randomly
            adjusted based on a normal distribution.

        use_action_obs (bool): Whether to use the action observation or not. NOTE: If
            the MjCambrianConstantActionWrapper is used, this is not reflected in the
            observation, as in the actions will vary in the observation.
        use_contact_obs (bool): Whether to use the contact observation or not. If this
            is True, then the contacts will be included in the observation space of the
            agent.

        eyes (Dict[str, MjCambrianEyeConfig]): The eyes on the agent. The keys are the
            names of the eyes and the values are the configs for the eyes. The eyes will
            be placed on the agent at the specified coordinates.
    """

    instance: Callable[[Self, str, int], "MjCambrianAgent"]

    # The name of the agent, optional.
    name: str = "Agent"

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
    perturb_init_pos: bool

    use_action_obs: bool
    use_contact_obs: bool

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
        self._config = config
        self._name = name

        self._eyes: Dict[str, MjCambrianEye] = {}

        self._spec: MjCambrianSpec = None
        self._init_pos: Tuple[float | None, float | None, float | None] = None
        self._init_quat: Tuple[
            float | None, float | None, float | None, float | None
        ] = None
        self._last_action: List[float] = None
        self._actuators: List[MjCambrianActuator] = []
        self._joints: List[MjCambrianJoint] = []
        self._geom: MjCambrianGeometry = None
        self._body: mj.MjsBody = None
        self._actadrs: List[int] = []
        self._body_id: int = None
        self._persistant_overlays: List[MjCambrianViewerOverlay] = []
        self._initialize()

    def _initialize(self):
        """Initialize the agent.

        This method does the following:
            - load the base xml to MjModel
            - parse the geometry
            - place eyes at the appropriate locations
        """
        try:
            spec = spec_from_xml_string(self._config.xml)
        except Exception:
            get_logger().error(f"Error creating model\n{self._config.xml}")
            raise

        self._parse_geometry(spec)
        self._parse_actuators(spec)

        self._create_eyes()

        assert len(self._config.init_pos) == 3, "init_pos must have 3 elements."
        self._init_pos = self._config.init_pos
        assert len(self._config.init_quat) == 4, "init_quat must have 4 elements."
        self._init_quat = self._config.init_quat

    def _parse_geometry(self, spec: MjCambrianSpec):
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
        model = spec.model

        # Num of controls
        if self.trainable:
            assert model.nu > 0, "Trainable agents must have controllable actuators."

        # Get number of qpos/qvel/ctrl
        # Just stored for later, like to get the observation space, etc.
        self._numqpos = model.nq
        self._numctrl = model.nu

        # Create the geometries we will use for eye placement
        geom_id = spec.get_geom_id(self._config.geom_name)
        assert geom_id != -1, f"Could not find geom {self._config.geom_name}."
        geom_rbound = model.geom_rbound[geom_id]
        geom_pos = model.geom_pos[geom_id]

        self._geom = MjCambrianGeometry(geom_id, geom_rbound, geom_pos)

    def _parse_actuators(self, spec: MjCambrianSpec):
        """Parse the current model/xml for the actuators.

        We have to do this twice: once on the initial model load to get the ctrl limits
        on the actuators (for the obs space). And then later to acquire the actual
        ids/adrs.
        """
        model = spec.model

        # Root body for the agent
        body_name = self._config.body_name
        body_id = spec.get_body_id(body_name)
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
            if (
                trntype == mj.mjtTrn.mjTRN_JOINT
                or trntype == mj.mjtTrn.mjTRN_JOINTINPARENT
            ):
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
                ctrllimited = model.actuator_ctrllimited[actadr]
                self._actuators.append(
                    MjCambrianActuator(actadr, trnid, ctrlrange, ctrllimited)
                )

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
        if self.trainable:
            assert len(self._actuators) > 0, f"Body {body_name} has no actuators."

    def _create_eyes(self):
        """Place the eyes on the agent."""
        for name, eye_config in self._config.eyes.items():
            self._eyes[name] = eye_config.instance(eye_config, f"{self._name}_{name}")
        # print(self._eyes.values())

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the agent. Will generate the xml from the model file
        and then add eyes to it.
        """
        xml = MjCambrianXML.from_string(self._config.xml)

        # Add eyes
        for eye in self.eyes.values():
            xml += eye.generate_xml(xml, self._geom, self._config.body_name)

        return xml

    def apply_action(self, actions: ActionType):
        """Applies the action to the agent. This probably happens before step
        so that the observations reflect the state of the agent after the new action
        is applied.

        It is assumed that the actions are normalized between -1 and 1.
        """
        self._last_action = actions.copy()
        if len(actions) == 0:
            return

        for action, actuator in zip(actions, self._actuators):
            if actuator.ctrllimited:
                action = np.interp(action, [-1, 1], actuator.ctrlrange)
            self._spec.data.ctrl[actuator.adr] = action

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

    def reset(self, spec: MjCambrianSpec) -> ObsType:
        """Sets up the agent in the environment. Uses the model/data to update
        positions during the simulation.
        """
        self._spec = spec
        self._persistant_overlays.clear()

        # Parse actuators; this is the second time we're doing this, but we need to
        # get the adrs of the actuators in the current model
        self._parse_actuators(spec)

        # Accumulate the qpos/qvel/act adrs
        self._reset_adrs(spec)

        # Reset the pose of the agent
        self._reset_pose(spec)

        # Set the last action to whatever the last ctrl was
        self._last_action = []
        for actuator in self._actuators:
            action = self._spec.data.ctrl[actuator.adr]
            if actuator.ctrllimited:
                action = np.interp(action, [-1, 1], actuator.ctrlrange)
            self._last_action.append(action)

        obs: Dict[str, Any] = {}
        for name, eye in self.eyes.items():
            eye_obs = eye.reset(spec)
            if isinstance(eye_obs, dict):
                obs.update(eye_obs)
            else:
                obs[name] = eye_obs

        return self._update_obs(obs)

    def _reset_adrs(self, spec: MjCambrianSpec):
        """Resets the adrs for the agent. This is used when the model is reloaded.

        .. todo::

            This was before the switch to spec. Is this still necessary?

        """

        # Root body for the agent
        body_name = self._config.body_name
        self._body_id = spec.get_body_id(body_name)
        assert self._body_id != -1, f"Could not find body with name {body_name}."

        # Geometry id
        geom_id = spec.get_geom_id(self._config.geom_name)
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

    def _reset_pose(self, spec: MjCambrianSpec):
        """Resets the pose of the agent."""
        self.pos = self._init_pos
        self.quat = self._init_quat

        # step here so that the states are updated
        mj.mj_forward(spec.model, spec.data)

        if self._config.perturb_init_pos:
            pos = np.random.normal(0, self._geom.rbound / 2, 2)
            pos = np.clip(pos, -self._geom.rbound / 2, self._geom.rbound / 2)
            self.pos += [*pos, 0]

    def step(self) -> ObsType:
        """Steps the eyes and returns the observation."""

        obs: ObsType = {}
        for name, eye in self.eyes.items():
            eye_obs = eye.step()
            if isinstance(eye_obs, dict):
                obs.update(eye_obs)
            else:
                obs[name] = eye_obs

        return self._update_obs(obs)

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Add additional attributes to the observation."""
        if self._config.use_action_obs:
            obs["action"] = self._last_action
        if self._config.use_contact_obs:
            obs["contacts"] = [self.has_contacts]

        return obs

    def render(self) -> list[MjCambrianViewerOverlay]:
        """Creates a list of overlays to render for the agent within the viewer."""
        if len(self._eyes) == 0:
            return []

        overlays = []
        create_text_overlay = MjCambrianViewerOverlay.create_text_overlay
        create_site_overlay = MjCambrianViewerOverlay.create_site_overlay

        # Add text overlays
        overlays.append(create_text_overlay(f"Agent: {self.name}"))
        action = f"Action: {', '.join([f'{a:.3f}' for a in self.last_action])}"
        overlays.append(create_text_overlay(action))
        if self._config.overlay_size > 0:
            rgba = self._config.overlay_color
            size = self._config.overlay_size
            if self.has_contacts:
                # Invert the color if there are contacts
                rgba = (1 - rgba[0], 1 - rgba[1], 1 - rgba[2], rgba[3])
            site_overlay = create_site_overlay(self.pos.copy(), rgba, size)
            self._persistant_overlays.append(site_overlay)

        # Add eye overlays
        for eye in self._eyes.values():
            overlays.extend(eye.render())

        overlays.extend(self._persistant_overlays)
        return overlays

    @property
    def has_contacts(self) -> bool:
        """Returns whether or not the agent has contacts.

        Walks through all the contacts in the environment and checks if any of them
        involve this agent.
        """
        if not self._config.check_contacts:
            return False

        for contact in self._spec.data.contact:
            if contact.exclude:
                continue

            geom1 = int(contact.geom[0])
            body1 = self._spec.model.geom_bodyid[geom1]
            rootbody1 = self._spec.model.body_rootid[body1]

            geom2 = int(contact.geom[1])
            body2 = self._spec.model.geom_bodyid[geom2]
            rootbody2 = self._spec.model.body_rootid[body2]

            is_this_agent = rootbody1 == self._body_id or rootbody2 == self._body_id
            if not is_this_agent or contact.exclude:
                # Not a contact with this agent
                continue

            return True

        return False

    @cached_property
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
            if isinstance(eye.observation_space, spaces.Dict):
                observation_space.update(eye.observation_space.spaces)
            else:
                observation_space[name] = eye.observation_space

        if self._config.use_action_obs:
            observation_space["action"] = self.action_space
        if self._config.use_contact_obs:
            observation_space["contacts"] = spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.int32
            )

        return spaces.Dict(observation_space)

    @cached_property
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
    def last_action(self) -> List[float]:
        return self._last_action

    @property
    def qpos(self) -> np.ndarray:
        """Gets the qpos of the agent. The qpos is the state of the joints defined
        in the agent's xml. This method is used to get the state of the qpos. It's
        actually a masked array where the entries are masked if the qpos adr is not
        associated with the agent. This allows the return value to be indexed
        and edited as if it were the full qpos array."""
        mask = np.ones(self._spec.data.qpos.shape, dtype=bool)
        mask[self._qposadrs] = False
        masked_qpos = np.ma.masked_array(self._spec.data.qpos, mask=mask)
        return masked_qpos[np.flatnonzero(~masked_qpos.mask)]

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
                self._spec.data.qpos[self._qposadrs[idx]] = val

    @property
    def pos(self) -> np.ndarray:
        """Returns the position of the agent in the environment.

        Note:
            the returned value, if edited, doesn't not directly impact the simulation.
            To set the position of the agent, use the `pos` setter.
        """
        return self._spec.data.xpos[self._body_id].copy()

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
                self._spec.data.qpos[self._qposadrs[idx]] = val

    @property
    def init_pos(self) -> Tuple[float | None, float | None, float | None]:
        """Returns the initial position of the agent in the environment."""
        assert self.body is not None, "Body is not set."
        return self.body.pos

    @init_pos.setter
    def init_pos(self, value: Tuple[float | None, float | None, float | None]):
        """Sets the initial position of the agent in the environment. The value is a
        tuple of the x, y, and z positions. If the value is None, the position is not
        updated."""
        init_pos = [None, None, None]
        for i in range(3):
            if self._config.init_pos[i] is not None:
                init_pos[i] = self._config.init_pos[i]
            elif i < len(value) and value[i] is not None:
                init_pos[i] = value[i]
            else:
                init_pos[i] = self._init_pos[i]
        self._init_pos = init_pos

    @property
    def quat(self) -> np.ndarray:
        """Returns the quaternion of the agent in the environment. Fmt: `wxyz`.

        Note:
            The returned value, if edited, doesn't not directly impact the simulation.
            To set the quaternion of the agent, use the `quat` setter.
        """
        return self._spec.data.xquat[self._body_id].copy()

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
                self._spec.data.qpos[self._qposadrs[3 + idx]] = val

    @property
    def trainable(self) -> bool:
        """Returns whether the agent is trainable or not."""
        return self._config.trainable

    @property
    def num_eyes(self) -> int:
        """Returns the number of eyes on the agent."""
        num_eyes = 0
        for eye in self.eyes.values():
            num_eyes += getattr(eye, "num_eyes", 1)
        return num_eyes


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
