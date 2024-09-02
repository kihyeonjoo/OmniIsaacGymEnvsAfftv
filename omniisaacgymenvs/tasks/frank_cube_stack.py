import math

import numpy as np
import torch

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

### Function to convert a 3D axis-angle representation of a rotation into a quaternion
# Not changed
@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

### CubeStack task class
class FrankaCubeStack(RLTask): # VecTask -> RLTask
    # Initionalization
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 23
        self._num_actions = 9

        RLTask.__init__(self, name, env)
        return    

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self._task_cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self._task_cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self._task_cfg["env"]["frankaDofNoise"]

        self.reward_settings = {
            "r_dist_scale": self._task_cfg["env"]["distRewardScale"],
            "r_lift_scale": self._task_cfg["env"]["liftRewardScale"],
            "r_align_scale": self._task_cfg["env"]["alignRewardScale"],
            "r_stack_scale": self._task_cfg["env"]["stackRewardScale"],
        }

    # get으로 객체를 불러오고 View를 통해 object을 scene에 내보내는 느낌. 
    # view로 한번에 관리함.
    def set_up_scene(self, scene) -> None:
        # reference에 쓰인 table 추가해야함
        self.get_franka()
        self.get_cubeA()
        self.get_cubeB()
        #self.get_table()
        #self.get_table_stand()
        
        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        #self._table = ~ -> RigidPrimView가 아니라 따로 View 만들어 줘야할듯? 외부에서 가져온거니까
        #self._table_stand = ~
        self._cubeA = RigidPrimView(
            prim_paths_expr="/World/envs/.*/prop/.*", name="cubeA_view", reset_xform_properties=False
        )
        self._cubeB = RigidPrimView(
            prim_paths_expr="/World/envs/.*/prop/.*", name="cubeB_view", reset_xform_properties=False
        )

        scene.add_ground_place()
        scene.add(self._frankas) 
        scene.add(self._frankas._hands) # franka_view
        scene.add(self._frankas._lfingers) # franka_view
        scene.add(self._frankas._rfingers) # franka_view
        #scene.add(self._table)
        #scene.add(self._table_stand)
        scene.add(self._cubeA) # 얘도 따로 설정해줘야하나? 굳이? prop이 있어서 상관없을듯?
        scene.add(self._cubeB) # 위와 같음

        # setup data
        self.init_data() # 이때 각 객체에 init_data 적용하는 듯
        return

    def initialize_views(self, scene):
        # reference에 쓰인 table 추가
        super().initialize_views(scene)
        if scene.object_exists("franka_view"):
            scene.remove_object("franka_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("drawers_view"):
            scene.remove_object("drawers_view", registry_only=True)
        if scene.object_exists("prop_view"):
            scene.remove_object("prop_view", registry_only=True)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    # get==crate: get_franka(완), get_table, get_table_stand, get_cubeA, get_cubeB)
    def get_franka(self):
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka")
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )
    def get_table(self):
        return
    def get_table_stand(self):
        return
    def get_cubeA(self): # cube A 생성
        drawer_pos = torch.tensor([0.0515, 0.0, 0.7172])
        prop_color = torch.tensor([0.6, 0.1, 0.0])
        prop_size = 0.05

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="cubeA",
            color=prop_color,
            size=prop_size,
            position=drawer_pos.numpy(),
            density=100.0
        )

        self._sim_config.apply_articulation_settings(
            "cubeA", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("cubeA")
        )
    def get_cubeB(self): # cube B 생성
        drawer_pos = torch.tensor([0.0515, 0.0, 0.7172])
        prop_color = torch.tensor([0.9, 0.4, 0.1])
        prop_size = 0.07

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="cubeB",
            color=prop_color,
            size=prop_size,
            position=drawer_pos.numpy(),
            density=100.0
        )

        self._sim_config.apply_articulation_settings(
            "cubeB", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("cubeB")
        )

    # scene 이후에 각 객체에 대해 init
    def init_data(self) -> None:
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

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")),
            self._device,
        )
        
        
        """
        ### finger_pose 설명
        finger_pose[0:3]은 lfinger_pose와 rfinger_pose position의 중앙으로 지정.
        finger_pose[3:7]은 lfinger_pose가 기준 회전점이 됨.
        finger_pose는 3개의 position([0:3]), 4개의 quaternion([3:7])을 가짐.
        """
        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]

        """
        ##### hand_pose_inv_rot, hand_pose_inv_pos 설명 #####
        각각 손의 회전의 역, 위치 역변환 좌표.
        손의 현재 위치와 회전 정보를 이용해 핑거의 상대적인 위치와 회전을 정의
        """
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        """
        ##### franka_local_grasp_pose_rot, franka_local_pose_pos 설명 #####
        franka_local_grashp_pose_rot: 손 기준으로 정의된 핑거의 회전 값.
        franka_local_pose_pos: 손 기준으로 정의된 핑거의 위치값.
        이를 통해 핑거 제어를 손 기준으로 정확하게 수행할 수 있게 도와줌.
        """
        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )

        """
        ##### franka_local_grasp_pos, frank_local_grasp_rot 설명 #####
        각각 repeat을 통해 조정된 위치와 회전을 여러 환경에 반복 적용하여 모든 환경에서 동일한 그립 동작이 이루어지도록 함.
        multi-gpu를 사용하여 학습을 하기 때문에 필요한 코드임. -> _num_envs: 몇개의 환경을 구성할 것인지.
        로봇의 그립이 물체를 잡기 위해 필요한 미세한 위치조정을 y축으로 0.04m 이동
        """
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        """
        ##### Start chainging code (drawer -> cube) #####
        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))
        ##### End chainging code #####

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        ##### Start chainging code (drawer -> cube) #####
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        ##### End chainging code #####
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        ##### Start chainging code (drawer -> cube) #####
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        ##### End chainging code #####

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)         
        """

        # cubeA의 포즈를 설정. 위치 [x, y, z]와 회전 [qw, qx, qy, qz] 값을 적절히 설정.
        cubeA_local_grasp_pose = torch.tensor([0.1, 0.1, 0.05, 1.0, 0.0, 0.0, 0.0], device=self._device)  # 예시 위치 및 회전값
        self.cubeA_local_grasp_pos = cubeA_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.cubeA_local_grasp_rot = cubeA_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        # 그리퍼가 물체를 향해 접근하거나 힘을 가하는 방향을 정의.
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        # cubeA의 그립 접근 방향을 설정합니다.
        self.cubeA_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        # 그리퍼의 위쪽을 정의, y축 방향을 기준으로 그리퍼의 기울기나 회전 방향을 설정하는데 사용
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        # cubeA의 위쪽 방향을 설정합니다.
        self.cubeA_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        # 로봇의 action 저장.
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        """
        cubeA에 대한 것도 더 설정해야하는지 알아볼 것. ~8/29
        """

    # observations을 통해 reward를 계산하기 위함
    def get_observations(self) -> dict:
        """
        ##### world 좌표계에서 pos, rot 가져오기
        hand_pos: 손(end effector) 위치, hand_rot: 손(end effector) 회전
        cubeA_pos: cubeA 위치, cubeA_rot: cubeA 회전
        """
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        cubeA_pos, cubeA_rot = self._cubeA.get_world_poses(clone=False) # scene 생성할때 _cubeA로 저장함. 정보 가져오는 코드.
        """
        ##### robot joint state 가져오기
        franka_dof_pos: each joint pos
        franka_dof_vel: each joint vel
        self.franka_dof_pos <- 내부 변수로 사용
        """
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.cubeA_dof_pos = self._cubeA.get_joint_positions(clone=False)
        self.cubeA_dof_vel = self._cubeA.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos
        self.cubeA_pos = cubeA_pos
        self.cubeA_rot = cubeA_rot
        ##### cube_pos, cube_rot 추가 필요 #####

        """
        ##### 로봇의 end effector와 물체의 pos, rot 정보를 바탕으로 그립 위치 및 회전 변환 계산
        self.franka_grasp_rot, self.franka_grasp_pos : 각각 end effector의 rot, pos
        self.cubeA_grasp_rot, self.cubeA_grasp_pos : 각각 cubeA의 grip rot, pos
        """
        (
            self.franka_grasp_rot,
            self.franka_grasp_pos,
            self.cubeA_grasp_rot,
            self.cubeA_grasp_pos,
        ) = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            cubeA_rot,
            cubeA_pos,
            self.cubeA_local_grasp_rot,
            self.cubeA_local_grasp_pos,
        )

        """
        self.franka_lfinger, self.franka_rfinger 설명
        각각 pos, rot 정보를 world에서 가져옴
        """
        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        """
        #### to_cubeA 설명
        """
        self.to_cubeA = self.cubeA_grasp_pos - self.franka_grasp_pos # to_cubeA = cubeA_pos - end effector pos-> 단순 상태인데 굳이 여기서 해야하나?

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                to_target,
                #self.cabinet_dof_pos[:, 3].unsqueeze(-1),
                #self.cabinet_dof_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )

        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations
    
    def pre_physics_step(self, actions) -> None:
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
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        ''' 
        삭제해도 될 것 같습니당

        # reset cabinet
        self._cabinets.set_joint_positions(
            torch.zeros_like(self._cabinets.get_joint_positions(clone=False)[env_ids]), indices=indices
        )
        self._cabinets.set_joint_velocities(
            torch.zeros_like(self._cabinets.get_joint_velocities(clone=False)[env_ids]), indices=indices
        )

        # reset props
        if self.num_props > 0:
            self._props.set_world_poses(
                self.default_prop_pos[self.prop_indices[env_ids].flatten()],
                self.default_prop_rot[self.prop_indices[env_ids].flatten()],
                self.prop_indices[env_ids].flatten().to(torch.int32),
            )
        '''

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        ### props 관련 코드라 cube로 대체해야 될 것 같습니다 ###
        if self.num_props > 0:
            self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
            self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
                self._num_envs, self.num_props
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    ### compute_franka_reward 함수 수정 후, 파라미터 수정 필요
    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.cabinet_dof_pos,
            self.franka_grasp_pos,
            self.drawer_grasp_pos,
            self.franka_grasp_rot,
            self.drawer_grasp_rot,
            self.franka_lfinger_pos,
            self.franka_rfinger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.distX_offset,
            self._max_episode_length,
            self.franka_dof_pos,
            self.finger_close_reward_scale,
        )

    ### 강화학습 종료 조건
    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )

    ### drawer -> cube로 수정 필요
    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        cubeA_rot,
        cubeA_pos,
        cubeA_local_grasp_rot,
        cubeA_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_cubeA_rot, global_cubeA_pos = tf_combine(
            cubeA_rot, cubeA_pos, cubeA_local_grasp_rot, cubeA_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_cubeA_rot, global_cubeA_pos

    """
    def compute_franka_reward(
        self,
        reset_buf,
        progress_buf,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        joint_positions,
        finger_close_reward_scale,
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)


        # axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        # axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        # axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        # axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        # dot1 = (
        #     torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # )  # alignment of forward axis for gripper
        # dot2 = (
        #     torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # )  # alignment of up axis for gripper
        # # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        # rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # # bonus if left finger is above the drawer handle and right below
        # around_handle_reward = torch.zeros_like(rot_reward)
        # around_handle_reward = torch.where(
        #     franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
        #     torch.where(
        #         franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2], around_handle_reward + 0.5, around_handle_reward
        #     ),
        #     around_handle_reward,
        # )
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward,
            ),
            finger_dist_reward,
        )

        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward
        )

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + around_handle_reward_scale * around_handle_reward
            + open_reward_scale * open_reward
            + finger_dist_reward_scale * finger_dist_reward
            - action_penalty_scale * action_penalty
            + finger_close_reward * finger_close_reward_scale
        )

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        # # prevent bad style in opening drawer
        # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards
    """
    @torch.jit.script
    def compute_franka_reward(self, states, reward_settings, m
    ):
        # Compute per-env physical parameters
        target_height = states["CubeB_size"] + states["cubeA_size"] / 2.0
        cubeA_size = states["cubeA_size"]
        
        # reward for distance from hand to the cubeA
        d = torch.norm(self.to_cubeA, dim=-1)
        d_lf = torch.norm(self.cubeA_pos - self.franka_lfinger_pos, dim=-1)
        d_rf = torch.norm(self.cubeA_pos - self.franka_rfinger_pos, dim=-1)
        dis_reward = 1- torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

        # reward for lifting cubeA
        cubeA_height = self.cubeA_pos[2] - 
        cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
        cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
        lift_reward = cubeA_lifted

