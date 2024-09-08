# 실행방법
# cd omniisaacgymenvs fordler
# copy and paste: C:\Users\kihye\AppData\Local\ov\pkg\isaac-sim-2023.1.1\python.bat scripts\rlgames_train.py task=FrankaPickandPlace headless=True

import numpy as np
import torch
# from isaacgym import gymtorch, gymapi
# from isaacgymenvs.tasks.base.rl_task import RLTask
# from isaacgymenvs.robots.articulations.franka import Franka
# from isaacgymenvs.robots.articulations.views.franka_view import FrankaView
# from omni.isaac.core.objects import DynamicCuboid
# from omni.isaac.core.prims import RigidPrimView
# from omni.isaac.core.utils.prims import get_prim_at_path

# core
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.articulations import Articulation, ArticulationView # jacobian때 불러오려고.
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *

### Modules for cube instead of cabinet
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply
# robots.articulations (franka, cubeA, cubeB)
from omniisaacgymenvs.robots.articulations.franka import Franka
# robots.articulations.views (franka, cubeA, cubeB)
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom

class FrankaPickAndPlaceTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        # Task configuration and parameters
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0
        
        self._num_observations = 23 # Observations include cube, gripper, and target positions
        self._num_actions = 9  # 7 for arm control, 2 for gripper control
        
        # Values to be filled in at runtime
        self.states = {}
        self.handles = {}
        self.num_dofs = None
        self.actions = None                             # Current actions to be deployed
        self._init_cube_state = None                    # Initial state of cube for the current env
        self._init_target_state = None                  # Initial state of target for the current env
        self._cube_state = None                         # Current state of cube for the current env
        self._target_state = None                       # Current state of target for the current env
        self._cube_id = None                            # Actor ID corresponding to cube for a given env
        self._target_id = None                          # Actor ID corresponding to target for a given env

        self._cube_size = 0.050                          # cube size
        self._target_size = 0.050                        # target size
        
        self._eef_state = None                          # Current state of end effector
        self._lf_state = None                       # Current state of left finger
        self._rf_state = None                       # Current state of left finger

        # Tensor placeholders
        self._j_eef = None                              # Jacobian for end effector using self.franka_articulation_view
        self._mm = None                                 # Mass matrix

        self._arm_control = None                        # Tensor buffer for contolling arm
        self._gripper_control = None                    # Tensor buffer for controlling gripper
        self._pos_control = None                        # Position actions
        self._effort_control = None                     # Torque actions

        self._franka_effort_limits = None               # Actuator effort limits for franka
        self._global_indices = None                     # Unique indices corresponding to all envs in flattened array -> ?

        # OSC Gains
        self.kp = None
        self.kd = None
        self.kp_null = None
        self.kd_null = None

        # Set control limits
        self.cmd_limit = None
        
        # Controller type
        self.control_type = None

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        # Update task configuration from simulation settings
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config


        # Controller type
        self.control_type = self._task_cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self._task_cfg["env"]["numObservations"] = 19 if self.control_type == "osc"  else 26
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self._task_cfg["env"]["numActions"] = 7  if self.control_type == "osc" else 8
        
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]

        self.reward_settings = {
            "r_dist_scale": self._task_cfg["env"]["distRewardScale"],
            "r_lift_scale": self._task_cfg["env"]["liftRewardScale"],
            "r_align_scale": self._task_cfg["env"]["alignRewardScale"],
            "r_stack_scale": self._task_cfg["env"]["stackRewardScale"],
        }

    def set_up_scene(self, scene) -> None:
        # Set up the scene with the robot and objects
        self.get_franka()
        self.get_cube()
        self.get_target()
        
        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._frankas_atc = ArticulationView(prim_paths_expr="/World/envs/.*/franka", name="franka_articulation_view")

        self._cubes = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view")
        self._target = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cubes)
        scene.add(self._target)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cubes = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view")
        self._target = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cubes)
        scene.add(self._target)

        self.init_data()

    def get_franka(self):
        # Load Franka robot
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka")
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )

    def get_cube(self):
        # Set up cube info
        cube_pos = torch.tensor([0.0515, 0.0, 0.7172])
        cube_color = torch.tensor([0.8, 0.2, 0.2])
        cube_size = self._cube_size
        
        # Load dynamic cube object
        cube = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/cube", name="cube",
            size=cube_size, 
            color=cube_color,
            position=cube_pos.numpy(),
            density=100.0
        )

        self._sim_config.apply_articulation_settings(
            "cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube")
        )

    def get_target(self):
        # Set up target info
        target_pos = torch.tensor([0.0915, 0.0, 0.7172])
        target_color = torch.tensor([0.2, 0.8, 0.2])
        target_size = self._target_size

        # Load target object as a fixed visual marker
        target = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/target", name="target",
            size=target_size,
            color=target_color,
            position=target_pos.numpy(), 
            density=0.0,
            #is_static=True
        )

    def init_data(self) -> None:
        self._compute_jacobian()

        # get environment local position
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)
        
        # get stage
        stage = get_current_stage()

        # GetPrimAtPath에서 불러오는 값들은 stage에 view를 통해 prim을 정해놓은 경로에서 불러오는 것.
        # cube, target은 initialize views 함수 참고
        '''
        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_link7", name="hands_view", reset_xform_properties=False
        )
        '''

        hand_state = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device,
        )
        self._lf_state = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device,
        )
        self._rf_state = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")),
            self._device,
        )
        cube_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/cube")),
            self._device,
        )
        target_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/target")),
            self._device,
        )

        # Get total DOFs 전체 자유도에 대한 값. 4096개의 env에 대한 자유도 계산
        self.num_dofs = 9

        # eef_state 선언 (px, py, pz, qw, qx, qy, qz)
        self._eef_state = torch.zeros(7, device=self._device)
        self._eef_state[0:3] = (self._lf_state[0:3] + self._rf_state[0:3]) / 2.0
        self._eef_state[3:7] = self._lf_state[3:7]
        


        # hand_pose 선언 (px, py, pz, qw, qx, qy, qz) ? 필요한가?

        # Initialize states
        self.states.update({
            "cube_size": torch.ones_like(self._eef_state[0]) * self._cube_size,
            "target_size": torch.ones_like(self._eef_state[0]) * self._target_size,
        })

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self._device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self._device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self._device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Initialize mass matrix
        self._mm = self._compute_mass_matrix()

        # Initialize actions
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        self._pos_control = torch.zeros((self._num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize simulation data for task
        self.franka_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        # dof_limits = self._frankas.get_dof_limits()
        # self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        # self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        # self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        # self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        # self.franka_dof_targets = torch.zeros((self._num_envs, self._frankas.num_dof), dtype=torch.float, device=self._device)

    def get_observations(self) -> dict:
        """기존에는 observation을 할 때 refresh function을 직접 개발해서 states들을 update하였으나,
        지금은 바로 get_world_poses로 pos, rot info 얻어옴. 즉, observation에서 바로 refresh되니까 state update도 여기서 바로 해줘야함.
        self.states에 값들 update하는 것도 Load 후에 하면 좋을듯? """

        ##### Load States #####
        # dof state
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        # hand state
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        # lfinger state
        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        # rfinger state
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._rfingers.get_world_poses(clonse=False)
        # cube state
        self.cube_pos, self.cube_rot = self._cubes.get_world_poses(clone=False)
        # target state
        self.target_pos, self.target_rot = self._target.get_world_poses

        # eef state
        self._eef_state[:3] = (self.franka_lfinger_pos + self.franka_rfinger_pos) / 2.0
        self._eef_state[3:7] = self.franka_lfinger_rot

        ##### Update States #####
        self.states.update({
            # Franka
            "q": franka_dof_pos,
            "q_gripper": franka_dof_pos[-2:],
            "eef_pos": self._eef_state[:3],
            "eef_rot": self._eef_state[3:7],
            "eef_lf_pos": self.franka_lfinger_pos,
            "eef_lf_rot": self.franka_lfinger_rot,
            "eef_rf_pos": self.franka_rfinger_pos,
            "eef_rf_rot": self.franka_rfinger_rot,
            # Cube
            "cube_pos": self.cube_pos,
            "cube_rot": self.cube_rot,
            # Target
            "target_pos": self.target_pos,
            "target_rot": self.target_rot,
            # relative pos (eef - cube)
            "cube_eef_pos": self.cube_pos - self._eef_state[:3],
            # cube to target pos
            "cube_target_pos": self.cube_pos - self.target_pos
        })

        ##### Update obs_buf #####
        """ cube_pos, cube_rot, cube_target_pos, eef_pos, eef_rot, q_gripper"""
        obs = ["cube_pos", "cube_rot", "cube_target_pos", "eef_pos", "eef_rot"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1),

        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations

        ##### 수정 필요 #####
    
    def pre_physics_step(self, actions) -> None:
        # Apply actions to the simulation
        if not self.world.is_playing():
            return

        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            ### _compute_osc_torques 수정(혹은 대체) 필요
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        ############## get_dof_limits 첫번째 차원이 env 수 인데, 어떤 env를 넣어야하는지? #######################
        # (u_gripper, upper_limit, lower_limit)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self._frankas_atc.get_dof_limits()[:, -2, 1],
                                      self._frankas_atc.get_dof_limits()[:, -2, 0])
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self._frankas_atc.get_dof_limits()[:, -1, 1],
                                      self._frankas_atc.get_dof_limits()[:, -1, 0])
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self._frankas_atc.set_joint_position_target(self._pos_control)
        self._frankas_atc.set_joint_efforts(self._effort_control)

    ##### 240904 1차 #####
    ##### ArticulationView의 jacobian 관련 함수로 대체 가능한지????
    def _compute_jacobian(self):
        """
        Computes the Jacobian for the end-effector of the Franka robot.
        """

        jacobian_tensors = self._frankas_atc.get_jacobians()
        jacobian = torch.tensor(jacobian_tensors)
        end_effector_joint_index = self._frankas_atc.get_dof_index("panda_joint7")
        self._j_eef = jacobian[:, :, : , end_effector_joint_index]

    def _compute_mass_matrix(self):
        """
        Computes the mass matrix for the Franka robot using ArticulationView in Isaac Sim
        """
        # returns (num_envs, num_dofs, num_dofs)
        mass_matrix_tensors = self._frankas_atc.get_mass_matrices()
        mass_matrices = torch.tensor(mass_matrix_tensors)
        # 7-DOF(body-DOF)에 대해서만 mass를 가져옴.
        mm = mass_matrices[:, :7, :7]
        return mm

    ##### 수정 필요 #####
    def post_physics_step(self):
        self.progress_buf += 1
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.get_observations()
        #### 사용되지 않는 parameter 정리 필요!!
        self.compute_franka_reward(self.states, self.reward_settings)

        # debug viz 생략
        
    ##### 수정 필요 #####
    def reset_idx(self, env_ids):
        # Reset specific environments
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # Reset Franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self._frankas.num_dof), device=self._device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        # Reset cube
        self._cubes.set_world_poses(
            torch.tensor([[0.0, 0.0, 0.05]], device=self._device).repeat((num_indices, 1)),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self._device).repeat((num_indices, 1)),
            indices
        )

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # Reset bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        ##### 필요한 인자들 선언 #####
        """
        q, qd, _j_eef
        """
    def _compute_osc_torques(self, dpose):
        q, qd = self._q[:7], self._qd[:7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
            self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)
        
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:7] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))
        
        return u

    def calculate_metrics(self) -> None:
        # Reward function for Pick-and-Place task
        # dist_to_target = torch.norm(self._cubes.get_world_poses(clone=False)[0] - self._targets.get_world_poses(clone=False)[0], dim=-1)
        # reward = -dist_to_target  # Simple reward based on distance to target
        
        self.rew_buf[:] = self.compute_franka_reward(
            self, self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self._max_episode_length
        )

        ##### 수정 필요 #####
    def is_done(self) -> None:
        # Termination condition: reset when the cube is close to the target or max steps reached
        # get_world_poses로 정보 얻어올 수 있음.
        dist_to_target = torch.norm(self._cubes.get_world_poses(clone=False)[0] - self._targets.get_world_poses(clone=False)[0], dim=-1)
        self.reset_buf = torch.where(dist_to_target < 0.05, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    # @torch.jit.script
    def compute_franka_reward(self, reset_buf, progress_buf, actions, states, reward_settings, max_episode_length):
        # Compute per-env physical parameters
        target_height = states["target_size"] + states["cube_size"] / 2.0
        cube_size = states["cube_size"]
        target_size = states["target_size"]

        # distance from hand to the cube
        d = torch.norm(states["cube_eef_pos"], dim=-1)
        d_lf = torch.norm(states["cube_pos"] - states["eef_lf_pos"], dim=-1)
        d_rf = torch.norm(states["cube_pos"] - states["eef_rf_pos"], dim=-1)
        dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

        # reward for  lifting cube
        cube_height = states["cube_pos"][2] - reward_settings["table_height"]
        cube_lifted = (cube_height - cube_size) > 0.04
        lift_reward = cube_lifted

        # how closely aligned cube is to target (only provided if cube is lifted)
        offset = torch.zeros_like(states["cube_target_pos"])
        offset[2] = (cube_size + target_size) / 2
        d_at = torch.norm(states["cube_target_pos"] + offset, dim=-1)
        align_reward = (1 - torch.tanh(10.0 * d_at)) * cube_lifted

        # Dist reward is maximum of dist and align reward
        dist_reward = torch.max(dist_reward, align_reward)

        # final reard for stacking successfully (only if cube is close to target height and corresponding location, and gripper is not grasping)
        cube_align_target = (torch.norm(states["cube_target_pos"][:2], dim=-1) < 0.02)
        cube_on_target = torch.abs(cube_height - target_height) < 0.02
        gripper_away_from_cube = (d>0.04)
        stack_reward = cube_align_target & cube_on_target & gripper_away_from_cube

        # We either provide the stack reward or  the align + dist reward
        rewards = torch.where(
            stack_reward,
            reward_settings["r_stack_scale"] * stack_reward, 
            reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + 
            reward_settings["r_align_scale"] * align_reward,
        )

        # Compute resets <- 이게 필요한가?
        # reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)

        return rewards