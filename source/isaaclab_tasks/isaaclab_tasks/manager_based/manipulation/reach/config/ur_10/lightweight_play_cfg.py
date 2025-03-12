# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
from isaaclab.sim import PhysxCfg

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import UR3_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR10ReachEnvCfg_UltraLight(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Minimal environment count for visualization
        self.scene.num_envs = 1  # Just a single environment
        self.scene.env_spacing = 2.0  # Reduced spacing since we have only one env

        # Low-quality physics simulation for performance
        self.sim = PhysxCfg(
            dt=0.02,  # Larger physics timestep
            substeps=1,  # Minimum substeps
            use_gpu_pipeline=False,  # Disable GPU physics
            use_fabric=False,  # Disable fabric
        )

        # switch robot to ur3
        self.scene.robot = UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        
        # override rewards - change ee_link to wrist_3_link
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["wrist_3_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]
        
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        
        # override command generator body - change ee_link to wrist_3_link
        self.commands.ee_pose.body_name = "wrist_3_link"
        # Keep existing pitch range configuration
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Disable all visualizations to improve performance
        self.visualizer.enable = False

        # Disable terrain and other complex scene elements
        self.scene.terrain.enable = False

        # Disable randomization for play to reduce compute
        self.observations.policy.enable_corruption = False 