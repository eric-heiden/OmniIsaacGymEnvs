# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.core.utils.prims import get_prim_at_path

import omniisaacgymenvs.tasks.cartpole_tiled as cartpole_tiled




import numpy as np
import torch
import math
import warp as wp
import warp.torch

@wp.kernel
def update_vbo(
    pole_world_positions: wp.array(dtype=wp.float32),
    pole_world_orientations: wp.array(dtype=wp.float32),
    cart_world_positions: wp.array(dtype=wp.float32),
    cart_world_orientations: wp.array(dtype=wp.float32),
    rails_world_positions: wp.array(dtype=wp.float32),
    num_envs : int,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4),
    vbo_orientations: wp.array(dtype=wp.quat)):

    tid = wp.tid()
    #shuffle quaternion layout w,x,y,z
    cart_pos = wp.vec4(cart_world_positions[tid*3]-rails_world_positions[tid*3],
                        cart_world_positions[tid*3+1]-rails_world_positions[tid*3+1],
                        cart_world_positions[tid*3+2]-rails_world_positions[tid*3+2],
                        1.0)
    cart_orn = wp.quat(cart_world_orientations[tid*4+1],cart_world_orientations[tid*4+2],cart_world_orientations[tid*4+3],cart_world_orientations[tid*4+0])
    
    
    pole_pos = wp.vec4(pole_world_positions[tid*3]-rails_world_positions[tid*3],
                        pole_world_positions[tid*3+1]-rails_world_positions[tid*3+1],
                        pole_world_positions[tid*3+2]-rails_world_positions[tid*3+2],
                        1.0)
    pole_orn = wp.quat(pole_world_orientations[tid*4+1],pole_world_orientations[tid*4+2],pole_world_orientations[tid*4+3],pole_world_orientations[tid*4+0])
    
    

    #rail indices start at 0, cart indices start at num_envs, pole indices start at num_envs*2
    vbo_positions[num_envs+tid] = cart_pos
    vbo_orientations[num_envs+tid] = cart_orn
    vbo_positions[2*num_envs+tid] = pole_pos
    vbo_orientations[2*num_envs+tid] = pole_orn


class CartpoleTiledCameraTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        print("self._env_spacing=",self._env_spacing)
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._tiled_cartpoles = cartpole_tiled.CartpoleTest(num_envs = self._num_envs , enable_tiled = True, use_matplot_lib = False)

        self._use_camera = True
        if self._use_camera:
            self._num_observations = self._tiled_cartpoles.camera_height * self._tiled_cartpoles.camera_width * self._tiled_cartpoles.camera_channels * self._tiled_cartpoles.camera_image_stack
            self._zero_obs = torch.zeros(self.num_envs*self._num_observations, device="cuda")
            self._zero_pos_vel = torch.zeros(self.num_envs*2, device="cuda")
            self._zero_pos_vel = torch.reshape(self._zero_pos_vel,(self._num_envs,2))
        else:
            self._num_observations = 4
        #print("self._num_observations=",self._num_observations)
        
        self.obs_buf2 = torch.zeros((self._num_envs, self.num_observations), device="cuda", dtype=torch.float)
        self._num_actions = 1

        RLTask.__init__(self, name, env)

        return

    def set_up_scene(self, scene) -> None:
        self.get_cartpole()
        super().set_up_scene(scene)
        self._cartpoles = ArticulationView(prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False)
        self._poles_view = RigidPrimView(prim_paths_expr="/World/envs/.*/Cartpole/pole", name="pole_view", reset_xform_properties=False)
        self._carts_view = RigidPrimView(prim_paths_expr="/World/envs/.*/Cartpole/cart", name="cart_view", reset_xform_properties=False)
        self._rails_view = RigidPrimView(prim_paths_expr="/World/envs/.*/Cartpole/rail", name="rail_view", reset_xform_properties=False)

        scene.add(self._cartpoles)
        scene.add(self._poles_view)
        scene.add(self._carts_view)
        scene.add(self._rails_view)

        return

    def get_cartpole(self):
        cartpole = Cartpole(prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole"))

    def sync_transforms(self, pole_world_positions, pole_world_orientations, cart_world_positions, cart_world_orientations, rails_world_positions):
        vbo = self._tiled_cartpoles.viz.opengl_app.cuda_map_vbo()
    
        num_instances = self._num_envs*3
        vbo_positions = wp.array(
        ptr=vbo.positions, dtype=wp.vec4, shape=(num_instances,),
        length=num_instances, capacity=num_instances,
        device="cuda", owner=False, ndim=1)
        vbo_orientations = wp.array(
        ptr=vbo.orientations, dtype=wp.quat, shape=(num_instances,),
        length=num_instances, capacity=num_instances,
        device="cuda", owner=False, ndim=1)

        
        wp_pole_world_positions = warp.from_torch(pole_world_positions.flatten())
        wp_pole_world_orientations = warp.from_torch(pole_world_orientations.flatten())
        wp_cart_world_positions = warp.from_torch(cart_world_positions.flatten())
        wp_cart_world_orientations = warp.from_torch(cart_world_orientations.flatten())
        wp_rails_world_positions = warp.from_torch(rails_world_positions.flatten())


        if 1:
            wp.launch(
            update_vbo,
            dim=self.num_envs,#not num_instances!
            inputs=[
                wp_pole_world_positions,
                wp_pole_world_orientations,
                wp_cart_world_positions,
                wp_cart_world_orientations,
                wp_rails_world_positions,
                self._num_envs,
            ],
            outputs=[
                vbo_positions,
                vbo_orientations,
            ],
            device="cuda")
        #print("vbo_positions=",vbo_positions)
        self._tiled_cartpoles.viz.opengl_app.cuda_unmap_vbo()


    def get_observations(self) -> dict:
        

        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)
        
        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        
        self.obs_buf2[:, 0] = cart_pos
        self.obs_buf2[:, 1] = cart_vel
        self.obs_buf2[:, 2] = pole_pos
        self.obs_buf2[:, 3] = pole_vel

        if self._use_camera:
            rails_world_positions, rails_world_orientations = self._rails_view.get_world_poses(clone=False)
            poles_world_positions, poles_world_orientations = self._poles_view.get_world_poses(clone=False)
            cart_world_positions, cart_world_orientations = self._carts_view.get_world_poses(clone=False)
            
            self.sync_transforms(poles_world_positions, poles_world_orientations, cart_world_positions, cart_world_orientations, rails_world_positions)
            self._tiled_cartpoles.update_observations()
            self.obs_buf = self._tiled_cartpoles.obs_buf
            observations = {
                self._cartpoles.name: {
                    #"obs_buf": self._tiled_cartpoles.obs_buf
                    #"obs_buf": self._zero_obs
                    "obs_buf": self.obs_buf
                }
            }
        else:
            observations = {
                self._cartpoles.name: {
                    "obs_buf": self.obs_buf2
                }
            }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        cart_pos = self.obs_buf2[:, 0]
        cart_vel = self.obs_buf2[:, 1]
        pole_angle = self.obs_buf2[:, 2]
        pole_vel = self.obs_buf2[:, 3]

        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        #print("reward=",reward)
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        cart_pos = self.obs_buf2[:, 0]
        pole_pos = self.obs_buf2[:, 2]

        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
