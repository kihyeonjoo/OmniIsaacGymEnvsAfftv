# 실행방법
# 

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

        self.cube_size = 0.050                          # cube size
        self.target_size = 0.050                        # target size
        
        self._eef_state = None

        # Tensor placeholders
        self._root_state = None         # 이런거 필요없는듯                # 
        self._dof_state = None


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

    def set_up_scene(self, scene) -> None:
        # Set up the scene with the robot and objects
        self.get_franka()
        self.get_cube()
        self.get_target()
        
        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cubes = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view")
        self._target = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view")

        scene.add(self._frankas)
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
        cube_size = 0.05
        
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
        target_size = 0.05

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
        # get environment local position
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputerLocalToWorldTransform(0)
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
        self._eef_lf_state = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device,
        )
        self._eef_rf_state = get_env_local_pose(
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
        #self.num_dofs = 

        # eef_state 선언 (px, py, pz, qw, qx, qy, qz)
        self._eef_state = torch.zeros(7, device=self._device)
        self._eef_state[0:3] = (self._eef_lf_state[0:3] + self._eef_rf_state[0:3]) / 2.0
        self._eef_state[3:7] = self._eef_lf_state[3:7]
        
        # Jacobian 선언 ()

        # hand_pose 선언 (px, py, pz, qw, qx, qy, qz)
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        # Initialize states
        self.states.update({
            "cube_size": torch.ones_like(self._eef_state[0]) * self.cube_size,
            "target_size": torch.ones_like(self._eef_state[0]) * self.target_size,
        })

        # Initialize actions
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        self._pos_control = torch.zeros((self._num_envs, self.num_dofs), dtype=torch.float, device=self._device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialzie control
        # self._arm_control = self._effort_control

        # Initialize simulation data for task
        self.franka_dof_pos = torch.zeros((self._num_envs, self._frankas.num_dof), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros((self._num_envs, self._frankas.num_dof), dtype=torch.float, device=self._device)

    def get_observations(self) -> dict:
        """기존에는 observation을 할 때 refresh function을 직접 개발해서 states들을 update하였으나,
        지금은 바로 get_world_poses로 pos, rot info 얻어옴. 즉, observation에서 바로 refresh되니까 state update도 여기서 바로 해줘야함.
        self.states에 값들 update하는 것도 Load 후에 하면 좋을듯? """

        ##### Load States #####
        # hand state
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        # lfinger state
        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        # rfinger state
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._rfingers.get_world_poses(clonse=False)
        # cube state
        cube_pos, cube_rot = self._cubes.get_world_poses(clone=False)
        # target state
        target_pos, target_rot = self._target.get_world_poses
        # dof state
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        # eef state
        eef_pos = self._eef_state[:3]
        eef_rot = self._eef_state[3:7]

        ##### Update States #####
        self.states.update({
            # Franka
            "eef_pos": self._eef_state[:3],
            "eef_rot": self._eef_state[3:7],
            "eef_lf_pos": self._eef_lf_state[:3]
        })

        ##### Calculate #####
        # cube_to_target_pos
        cube_to_target_pos = torch.norm(cube_pos - target_pos)

        ##### Update obs_buf #####
        """ cube_pos, cube_rot, cube_to_target_pos, eef_pos, eef_rot, q_gripper"""
        self.obs_buf = torch.cat(
            (
                cube_pos,
                cube_rot,
                cube_to_target_pos,
                eef_pos,
                eef_rot,
                ### What is q_gripper??? ###
            ),
            dim=-1,
        )

        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        # Apply actions to the simulation
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

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

    def calculate_metrics(self) -> None:
        # Reward function for Pick-and-Place task
        dist_to_target = torch.norm(self._cubes.get_world_poses(clone=False)[0] - self._targets.get_world_poses(clone=False)[0], dim=-1)
        reward = -dist_to_target  # Simple reward based on distance to target
        
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        # Termination condition: reset when the cube is close to the target or max steps reached
        # get_world_poses로 정보 얻어올 수 있음.
        dist_to_target = torch.norm(self._cubes.get_world_poses(clone=False)[0] - self._targets.get_world_poses(clone=False)[0], dim=-1)
        self.reset_buf = torch.where(dist_to_target < 0.05, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    
    # Compute per-env physical parameters
    target_height = states["target_size"] + states["cube_size"] / 2.0
    cube_size = states["cube_size"]
    target_size = states["target_size"]

    # distance from hand to the cube
    d = torch.norm(states["cube_pos_relative"], dim=-1)
    d_lf = torch.norm(states["cube_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cube_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # reward for  lifting cube
    cube_height = states["cube_pos"][2] - reward_settings["table_height"]
    cube_lifted = (cube_height - cube_size) > 0.04
    lift_reward = cube_lifted

    # how closely aligned cube is to target (only provided if cube is lifted)
    offset = torch.zeros_like(states["cube_to_target_pos"])
    offset[2] = (cube_size + target_size) / 2
    d_at = torch.norm(states["cube_to_target_pos"] + offset, dim=-1)
    align_reward = (1 - torch.tanh(10.0 * d_at)) * cube_lifted

    # Dist reward is maximum of dist and align reward
    dist_reward = torch.max(dist_reward, align_reward)

    # final reard for stacking successfully (only if cube is close to target height and corresponding location, and gripper is not grasping)
    cube_align_target = (torch.norm(states["cube_to_target_pos"][:2], dim=-1) < 0.02)
    cube_on_target = torch.abs(cube_height - target_height) < 0.02
    gripper_away_from_cube = (d>0.04)
    stack_reward = cube_align_cube & cube_on_target & gripper_away_from_cube

    # We either provide the stack reward or  the align + dist reward
    rewards = torch.where(
        stack_reward,
        reward_settings["r_stack_scale"] * stack_reward, 
        reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + 
        reward_settings["r_align_scale"] * align_reward,
    )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf