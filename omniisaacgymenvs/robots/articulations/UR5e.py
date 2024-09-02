import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema

class UR5e(Robot):
    def __init__(
            self,
            prim_path: str,
            name: Optional[str] = "ur5e",
            usd_path: Optional[str] = None,
            translation: Optional[torch.tensor] = None,
            orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Issac Sim assets folder")
            self._usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
            # we need to modify _usd_path to our local path

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__( 
            prim_path=prim_path,
            name=name,
            trainslation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths =[
            
        ]

        drive_type = ["angular"] * 7 + ["linear"] * 2
        # modify _dof_info
        default_dof_pos = [math.degress(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0.02, 0.02]
        stiffness = [400 * np.pi / 180] * 7 + [10000] * 2
        damping = [80 * np.pi / 180] * 7 + [100] * 2
        max_force = [87, 87, 87, 87, 12, 12, 12, 200, 200]
        max_velocity = [math.degress(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] + [0.2, 0.2]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )

    def set_ur5e_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhyscRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)