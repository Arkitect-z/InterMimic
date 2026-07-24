from enum import Enum
import csv
import json
import math
import numpy as np
import torch
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from ...utils import torch_utils
import torch.nn.functional as F
from .humanoid import *
import trimesh

from ...utils.path_utils import resolve_data_path



class InterMimic(Humanoid_SMPLX):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._use_gpu_pipeline = bool(sim_params.use_gpu_pipeline)
        state_init = cfg["env"]["stateInit"]
        self._state_init = InterMimic.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']
        self.ball_size = cfg['env']['ballSize']
        self.more_rigid = cfg['env']['moreRigid']
        self.rollout_length = cfg['env']['rolloutLength']
        self.psi = cfg['env'].get('physicalBufferSize', 1)
        self.fps_data = float(cfg['env'].get('dataFPS', 30))
        self._correct_contact_distance = float(cfg['env'].get('correctContactDistance', 0.04))
        self._reference_angular_velocity_limit = float(
            cfg['env'].get('referenceAngularVelocityLimit', 20.0)
        )
        # Contact labels are useful as a weak imitation prior, but they are not
        # a task-success definition.  In particular, converted multi-sequence
        # data can be off by several frames and a successful policy may use a
        # different (but physically valid) grasp.  Keep the legacy behavior
        # available for checkpoint reproduction while allowing outcome-driven
        # training to avoid hard GT-contact constraints.
        self._contact_reward_mode = str(
            cfg['env'].get('contactRewardMode', 'legacy_multiplicative')
        ).lower()
        if self._contact_reward_mode not in {
            'legacy_multiplicative', 'soft', 'none'
        }:
            raise ValueError(
                "contactRewardMode must be one of "
                "{legacy_multiplicative, soft, none}, got "
                f"{self._contact_reward_mode!r}"
            )
        self._enable_contact_failure_termination = bool(
            cfg['env'].get('enableContactFailureTermination', False)
        )
        self._enable_wrist_failure_termination = bool(
            cfg['env'].get('enableWristFailureTermination', True)
        )
        self._enable_object_contact_phase_termination = bool(
            cfg['env'].get('enableObjectContactPhaseTermination', True)
        )
        self._contact_failure_grace_frames = int(
            cfg['env'].get('contactFailureGraceFrames', 20)
        )
        if self._contact_failure_grace_frames < 0:
            raise ValueError("contactFailureGraceFrames must be non-negative")
        # Evaluation only works with stateInit "Start"
        state_init_is_start = (state_init == "Start")
        self.enable_evaluation = cfg['env'].get('enableEvaluation', False) and state_init_is_start
        if cfg['env'].get('enableEvaluation', False) and not state_init_is_start:
            print(f"Warning: Evaluation is disabled because stateInit is '{state_init}' (must be 'Start')")
        motion_entries = os.listdir(self.motion_file)
        data_subsets = cfg['env'].get('dataSub', ['*'])
        include_all = not data_subsets or any(
            str(value).lower() in {'*', 'all'} for value in data_subsets
        )
        self.motion_file = sorted([
            os.path.join(self.motion_file, data_path)
            for data_path in motion_entries
            if data_path.endswith('.pt')
            and (
                include_all
                or data_path.split('_', 1)[0] in data_subsets
            )
        ])
        if not self.motion_file:
            raise ValueError(
                f"No .pt motion files selected from {cfg['env']['motion_file']} "
                f"by dataSub={data_subsets}"
            )

        # Parse dual-object names from filename: sub1_ObjA+ObjB_seqname.pt
        self._motion_obj_pairs = []  # [(obj1, obj2)] per motion
        unique_obj_set = set()
        for path in self.motion_file:
            name_tokens = os.path.basename(path).rsplit('.', 1)[0].split('_')
            object_tokens = [token for token in name_tokens if '+' in token]
            if len(object_tokens) != 1:
                raise ValueError(
                    f"Motion filename must contain exactly one ObjA+ObjB token: {path}"
                )
            combined = object_tokens[0]
            if '+' in combined:
                o1, o2 = combined.split('+', 1)
            else:
                o1, o2 = combined, combined
            if not o1 or not o2:
                raise ValueError(f"Invalid object-pair token '{combined}' in {path}")
            self._motion_obj_pairs.append((o1, o2))
            unique_obj_set.update([o1, o2])
        if int(cfg['env']['numEnvs']) < len(self._motion_obj_pairs):
            raise ValueError(
                f"numEnvs={cfg['env']['numEnvs']} cannot cover "
                f"{len(self._motion_obj_pairs)} motion sequences"
            )

        # Construct device string before super().__init__()
        if device_type == "cuda" or device_type == "GPU":
            self._init_device = "cuda:" + str(device_id)
        else:
            self._init_device = "cpu"

        self.object_name = sorted(list(unique_obj_set))  # unique object types
        self._motion_ids = torch.arange(
            len(self._motion_obj_pairs), device=self._init_device, dtype=torch.long
        )
        self.obj1_id = to_torch(
            [self.object_name.index(p[0]) for p in self._motion_obj_pairs],
            dtype=torch.long,
            device=self._init_device,
        )
        self.obj2_id = to_torch(
            [self.object_name.index(p[1]) for p in self._motion_obj_pairs],
            dtype=torch.long,
            device=self._init_device,
        )
        # Backward compat: object_id points to obj1 for legacy code paths
        self.object_id = self.obj1_id
        # With dual objects, every motion contains both objects — all motions are valid
        self.obj2motion = torch.ones((len(self.object_name), len(self._motion_obj_pairs)), dtype=torch.bool).to(self._init_device)
        self.robot_type = cfg['env']['robotType']
        self.object_density = cfg['env']['objectDensity']
        self._require_object_density = bool(
            cfg['env'].get('requireObjectDensity', False)
        )
        # 2-object hoi_data: human(7+306+676) + obj1(13) + obj2(13) + ig1(156) + ig2(156) + contact(52+1+1)
        self.ref_hoi_obs_size = 7 + 51 * 6 + 52 * 13 + 13 * 2 + 52 * 3 * 2 + 52 + 2
        self.num_motions = len(self.motion_file)
        dataset_indices = []
        for data_path in self.motion_file:
            subset = os.path.basename(data_path).split('_', 1)[0]
            if not subset.startswith('sub') or not subset[3:].isdigit():
                raise ValueError(
                    f"Motion filename must start with sub<number>_: {data_path}"
                )
            dataset_indices.append(int(subset[3:]))
        self.dataset_index = to_torch(
            dataset_indices, dtype=torch.long, device=self._init_device
        )

        self._preload_table_info()

        if self.play_dataset:
            sim_params.gravity = gymapi.Vec3(0, 0, 0)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._resolve_control_dof_indices()
        self._motion_obj_reset_thresholds = to_torch(
            [
                [
                    info.get('reset_dist', 0.05) if info is not None else 0.05
                    for info in motion_info
                ]
                for motion_info in self._motion_table_info
            ],
            device=self.device,
            dtype=torch.float,
        )

        # Reduce contact_offset on arm/hand bodies for tighter hand-object contact
        _arm_hand_names = [
            'L_Shoulder', 'L_Elbow', 'L_Wrist',
            'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
            'R_Shoulder', 'R_Elbow', 'R_Wrist',
            'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3',
        ]
        for env_ptr, handle in zip(self.envs, self.humanoid_handles):
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            body_names = self.gym.get_actor_rigid_body_names(env_ptr, handle)
            for b_idx, bname in enumerate(body_names):
                if bname in _arm_hand_names:
                    props[b_idx].contact_offset = 0.001
                    props[b_idx].rest_offset = 0.0
                    props[b_idx].friction = 1.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

        if self.play_dataset:
            for env_ptr, handle in zip(self.envs, self.humanoid_handles):
                dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
                dof_prop['stiffness'][:] = 0
                dof_prop['damping'][:] = 0
                dof_prop['driveMode'][:] = gymapi.DOF_MODE_NONE
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        self.hoi_data = self._load_motion(self.motion_file, topk=self.psi)

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.kinematic_reset = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.contact_reset = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self._hand_fail_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._obj_fail_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._contact_fail_counter = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self._hand_fail_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._diag_file = None
        if self.num_envs <= 16:
            diag_path = os.path.join(os.getcwd(), 'diag_output.txt')
            self._diag_file = open(diag_path, 'w')
            print(f"[DIAG] Writing per-step diagnostics to {diag_path}")
        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # PSI is disabled when physicalBufferSize <= 1.  Do not reserve the
        # ~0.8 GiB state buffer in the normal Theia training/evaluation path.
        self._curr_reward = None
        self._sum_reward = None
        self._curr_state = None
        if self.psi > 1:
            buf_len = max(cfg['env']['rolloutLength'], cfg['env']['episodeLength'])
            self._curr_reward = torch.zeros([self.num_envs, buf_len], device=self.device, dtype=torch.float)
            self._sum_reward = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
            self._curr_state = torch.zeros([self.num_envs, buf_len, 345], device=self.device, dtype=torch.float)
        self._build_target_tensors()
        if cfg['env'].get('validateReferenceFK', False):
            self._validate_reference_fk()

        return

    def _valid_motions_for_env(self, env_id):
        """Each environment is bound to one sequence and its fixed tables."""
        env_idx = int(env_id.item()) if torch.is_tensor(env_id) else int(env_id)
        return self._motion_ids[env_idx % len(self._motion_obj_pairs)].view(1)

    def _sample_motion_for_env(self, env_id):
        valid = self._valid_motions_for_env(env_id)
        return valid[torch.randint(valid.numel(), (), device=self.device)]

    def _resolve_control_dof_indices(self):
        """Resolve critical control groups by asset DOF name and fail fast."""
        fallback = {
            'L_Wrist_x': 48, 'L_Wrist_y': 49, 'L_Wrist_z': 50,
            'R_Wrist_x': 105, 'R_Wrist_y': 106, 'R_Wrist_z': 107,
        }
        if hasattr(self.gym, 'get_actor_dof_names'):
            names = list(self.gym.get_actor_dof_names(self.envs[0], self.humanoid_handles[0]))
            name_to_idx = {name: idx for idx, name in enumerate(names)}
            missing = [name for name in fallback if name not in name_to_idx]
            if missing:
                raise RuntimeError(f"Missing expected wrist DOFs in humanoid asset: {missing}")
            resolved = [name_to_idx[name] for name in fallback]
        else:
            resolved = list(fallback.values())

        expected = list(fallback.values())
        if resolved != expected:
            raise RuntimeError(f"Unexpected wrist DOF layout: resolved={resolved}, expected={expected}")
        self._wrist_dof_idx = resolved

    def _validate_reference_fk(self):
        """Validate every PT motion against its bound Isaac articulation."""
        validation_offset = torch.tensor(
            [[0.0, 0.0, 5.0]], device=self.device, dtype=torch.float
        )
        position_errors = []
        rotation_errors = []
        sample_metadata = []

        # Run the articulation above collision geometry and without gravity so
        # the propagation step measures FK rather than contacts or free-fall.
        sim_params = self.gym.get_sim_params(self.sim)
        original_gravity = gymapi.Vec3(
            sim_params.gravity.x, sim_params.gravity.y, sim_params.gravity.z
        )
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.set_sim_params(self.sim, sim_params)

        try:
            # Environment i is bound to motion i, including its object pair
            # and support geometry. Validate all motions in parallel so a
            # thousand-sequence dataset still costs only eight PhysX steps.
            env_ids = torch.arange(
                self.num_motions, device=self.device, dtype=torch.long
            )
            motion_ids = env_ids.clone()
            ref_ids = torch.zeros_like(motion_ids)
            frame_limits = self.max_episode_length[motion_ids] - 1

            for fraction in torch.linspace(
                0.0, 1.0, steps=8, device=self.device
            ):
                times = torch.round(
                    frame_limits.float() * fraction
                ).long()
                self.data_id[env_ids] = motion_ids
                self.ref_index[env_ids] = ref_ids
                self.progress_buf[env_ids] = times
                root_pos = self.extract_ref_component(
                    'root_pos', motion_ids, ref_ids, times
                )
                self._set_env_state(
                    env_ids=env_ids,
                    root_pos=root_pos + validation_offset,
                    root_rot=self.extract_ref_component(
                        'root_rot', motion_ids, ref_ids, times
                    ),
                    dof_pos=self.extract_ref_component(
                        'dof_pos', motion_ids, ref_ids, times
                    ),
                    root_vel=torch.zeros(
                        (self.num_motions, 3), device=self.device
                    ),
                    root_ang_vel=torch.zeros(
                        (self.num_motions, 3), device=self.device
                    ),
                    dof_vel=torch.zeros(
                        (self.num_motions, self.num_dof),
                        device=self.device,
                    ),
                )
                self._reset_target(env_ids)
                self._reset_env_tensors(env_ids)
                # Keep the tensor alive until PhysX consumes the pointer
                # returned by gymtorch.unwrap_tensor.
                pd_targets = self._dof_pos.clone()
                self.gym.set_dof_position_target_tensor(
                    self.sim, gymtorch.unwrap_tensor(pd_targets)
                )
                # Indexed setters do not update rigid-body state until a
                # simulation step has propagated the articulation.
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self._refresh_sim_tensors()

                ref_obs = self.hoi_data[motion_ids, times]
                ref_pos = self.extract_data_component(
                    'body_pos', obs=ref_obs
                ).view(self.num_motions, 52, 3)
                ref_pos = ref_pos + validation_offset.unsqueeze(1)
                ref_rot = self.extract_data_component(
                    'body_rot', obs=ref_obs
                ).view(self.num_motions, 52, 4)
                position_errors.append(
                    (
                        self._rigid_body_pos[env_ids] - ref_pos
                    ).norm(dim=-1)
                )
                diff = torch_utils.quat_mul_norm(
                    torch_utils.quat_inverse(
                        ref_rot.reshape(-1, 4)
                    ),
                    self._rigid_body_rot[env_ids].reshape(-1, 4),
                )
                angle, _ = torch_utils.quat_to_angle_axis(diff)
                rotation_errors.append(
                    angle.view(self.num_motions, 52)
                )
                sample_metadata.extend(zip(
                    motion_ids.tolist(), times.tolist()
                ))
        finally:
            sim_params.gravity = original_gravity
            self.gym.set_sim_params(self.sim, sim_params)

        pos_error = torch.cat(position_errors, dim=0)
        rot_error = torch.cat(rotation_errors, dim=0)
        if not torch.isfinite(pos_error).all():
            raise RuntimeError(
                "Reference FK validation produced non-finite position errors"
            )
        if not torch.isfinite(rot_error).all():
            raise RuntimeError(
                "Reference FK validation produced non-finite rotation errors"
            )
        mean_pos = float(pos_error.mean().item())
        max_pos = float(pos_error.max().item())
        mean_rot_deg = float(torch.rad2deg(rot_error).mean().item())
        max_rot_deg = float(torch.rad2deg(rot_error).max().item())
        print(
            f"[VALIDATE] Isaac FK vs PT: motions={self.num_motions}, "
            f"mean_pos={mean_pos:.6f}m, max_pos={max_pos:.6f}m, "
            f"mean_rot={mean_rot_deg:.3f}deg, "
            f"max_rot={max_rot_deg:.3f}deg"
        )
        body_names = list(
            self.gym.get_actor_rigid_body_names(
                self.envs[0], self.humanoid_handles[0]
            )
        )
        for label, errors in [
            ("position", pos_error),
            ("rotation_deg", torch.rad2deg(rot_error)),
        ]:
            values, flat_indices = torch.topk(
                errors.flatten(), k=min(5, errors.numel())
            )
            diagnostics = []
            for value, flat_idx in zip(
                values.tolist(), flat_indices.tolist()
            ):
                sample_idx = flat_idx // errors.shape[1]
                body_idx = flat_idx % errors.shape[1]
                motion_idx, frame = sample_metadata[sample_idx]
                diagnostics.append(
                    f"motion={motion_idx} f={frame} "
                    f"body={body_names[body_idx]} value={value:.6f}"
                )
            print(f"[VALIDATE] worst {label}: " + "; ".join(diagnostics))
        pos_tol = float(self.cfg['env'].get('fkMaxPositionError', 0.005))
        rot_tol = float(
            self.cfg['env'].get('fkMaxRotationErrorDeg', 1.0)
        )
        if max_pos > pos_tol or max_rot_deg > rot_tol:
            raise RuntimeError(
                f"Reference FK validation failed: max_pos={max_pos:.6f}m "
                f"(tol={pos_tol}), max_rot={max_rot_deg:.3f}deg "
                f"(tol={rot_tol})"
            )

    def post_physics_step(self):
        super().post_physics_step()
        return

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def _preload_table_info(self):
        """Compute per-motion support tables from each frame-0 object pose."""
        # _table_info stores one geometry template per object asset, while
        # _motion_table_info keeps the sequence-specific support pose.
        self._table_info = {}
        self._motion_table_info = []
        from scipy.spatial.transform import Rotation as R

        if not self.motion_file:
            return
        for data_path, pair in zip(self.motion_file, self._motion_obj_pairs):
            data = torch.load(data_path, weights_only=False)
            obj_specs = [
                (pair[0], 318, 321, 325),
                (pair[1], 325, 328, 332),
            ]
            motion_info = []
            for obj_name, pos_s, pos_e, rot_e in obj_specs:
                obj_pos_0 = data[0, pos_s:pos_e].numpy()
                obj_rot_0 = data[0, pos_e:rot_e].numpy()

                obj_file = resolve_data_path(
                    "assets", "objects", "objects", obj_name, obj_name + ".obj"
                )
                if not os.path.exists(str(obj_file)):
                    raise FileNotFoundError(
                        f"Missing object mesh required by {data_path}: {obj_file}"
                    )
                mesh = trimesh.load(str(obj_file), force='mesh')
                rot = R.from_quat(obj_rot_0)
                verts_world = rot.apply(mesh.vertices) + obj_pos_0
                info = {
                    'table_top_z': float(verts_world[:, 2].min()),
                    'half_x': max(mesh.extents[0] * 0.5, 0.03),
                    'half_y': max(mesh.extents[1] * 0.5, 0.03),
                    'init_x': float(obj_pos_0[0]),
                    'init_y': float(obj_pos_0[1]),
                    'reset_dist': float(mesh.extents.max() * 0.3),
                }
                motion_info.append(info)
                self._table_info.setdefault(obj_name, info)
            self._motion_table_info.append(tuple(motion_info))

    def _clip_reference_angular_velocity(self, velocity):
        norm = velocity.norm(dim=-1, keepdim=True)
        scale = torch.clamp(
            self._reference_angular_velocity_limit / torch.clamp(norm, min=1e-8),
            max=1.0,
        )
        return velocity * scale

    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):

        hoi_datas = []
        hoi_refs = []
        motion_requires_simultaneous_grasp = []
        if type(motion_file) != type([]):
            motion_file = [motion_file]
        max_episode_length = []
        # Process data on CPU first, then move to GPU at the end
        object_points_cpu = self.object_points.cpu()
        object_id_cpu = self.object_id.cpu()

        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = torch.load(
                data_path, weights_only=False
            )[startk:]
            if hoi_data.ndim != 2 or hoi_data.shape[1] != 594:
                raise ValueError(
                    f"Unsupported Theia motion schema in {data_path}: "
                    f"expected [T, 594], got {tuple(hoi_data.shape)}"
                )
            if not torch.isfinite(hoi_data).all():
                raise ValueError(f"Non-finite values in Theia motion: {data_path}")
            loaded_dict['hoi_data'] = hoi_data.detach()  # Keep on CPU for processing


            max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()
            loaded_dict['root_rot_vel'] = torch_utils.quat_sequence_angular_velocity(
                loaded_dict['root_rot'], self.fps_data
            )
            loaded_dict['root_rot_vel'] = self._clip_reference_angular_velocity(
                loaded_dict['root_rot_vel']
            )

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone()

            loaded_dict['dof_vel'] = []

            loaded_dict['dof_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_vel'].shape[-1])),loaded_dict['dof_vel']),dim=0)

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162+52*3].clone()
            loaded_dict['body_pos_vel'] = (loaded_dict['body_pos'][1:,:].clone() - loaded_dict['body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_pos_vel'].shape[-1])),loaded_dict['body_pos_vel']),dim=0)

            # --- Object 1 ---
            loaded_dict['obj1_pos'] = loaded_dict['hoi_data'][:, 318:321].clone()
            loaded_dict['obj1_pos_vel'] = torch.cat((torch.zeros((1, 3)), (loaded_dict['obj1_pos'][1:] - loaded_dict['obj1_pos'][:-1]) * self.fps_data), dim=0)
            loaded_dict['obj1_rot'] = loaded_dict['hoi_data'][:, 321:325].clone()
            loaded_dict['obj1_rot_vel'] = torch_utils.quat_sequence_angular_velocity(
                loaded_dict['obj1_rot'], self.fps_data
            )
            loaded_dict['obj1_rot_vel'] = self._clip_reference_angular_velocity(
                loaded_dict['obj1_rot_vel']
            )

            # --- Object 2 ---
            loaded_dict['obj2_pos'] = loaded_dict['hoi_data'][:, 325:328].clone()
            loaded_dict['obj2_pos_vel'] = torch.cat((torch.zeros((1, 3)), (loaded_dict['obj2_pos'][1:] - loaded_dict['obj2_pos'][:-1]) * self.fps_data), dim=0)
            loaded_dict['obj2_rot'] = loaded_dict['hoi_data'][:, 328:332].clone()
            loaded_dict['obj2_rot_vel'] = torch_utils.quat_sequence_angular_velocity(
                loaded_dict['obj2_rot'], self.fps_data
            )
            loaded_dict['obj2_rot_vel'] = self._clip_reference_angular_velocity(
                loaded_dict['obj2_rot_vel']
            )

            # --- IG (SDF) for both objects ---
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, 52, 1).view(-1, 4)
            body_pos_3d = loaded_dict['body_pos'].view(max_episode_length[-1], 52, 3)

            obj1_id_cpu = self.obj1_id.cpu()
            obj2_id_cpu = self.obj2_id.cpu()
            for obj_key, obj_rot_key, obj_pos_key, obj_id in [
                ('ig1', 'obj1_rot', 'obj1_pos', obj1_id_cpu),
                ('ig2', 'obj2_rot', 'obj2_pos', obj2_id_cpu),
            ]:
                o_rot = loaded_dict[obj_rot_key]
                o_pos = loaded_dict[obj_pos_key]
                pts = object_points_cpu[obj_id[idx]]
                o_rot_ext = o_rot.unsqueeze(1).repeat(1, pts.shape[0], 1).view(-1, 4)
                pts_ext = pts.unsqueeze(0).repeat(o_rot.shape[0], 1, 1).view(-1, 3)
                o_pts = torch_utils.quat_rotate(o_rot_ext, pts_ext).view(o_rot.shape[0], pts.shape[0], 3) + o_pos.unsqueeze(1)
                ig = compute_sdf(body_pos_3d, o_pts).view(-1, 3)
                ig = quat_rotate(heading_rot_extend, ig).view(o_rot.shape[0], -1)
                loaded_dict[obj_key] = ig

            loaded_dict['contact_obj1'] = torch.round(loaded_dict['hoi_data'][:, 332:333].clone())
            loaded_dict['contact_obj2'] = torch.round(loaded_dict['hoi_data'][:, 333:334].clone())
            loaded_dict['contact_human'] = torch.round(loaded_dict['hoi_data'][:, 334:334+52].clone())
            stable_frames = int(
                self.cfg['env'].get('evaluationStableFrames', 10)
            )
            left_contact = (
                loaded_dict['contact_human'][:, 17:33] > 0.5
            ).any(dim=-1)
            right_contact = (
                loaded_dict['contact_human'][:, 36:52] > 0.5
            ).any(dim=-1)
            simultaneous = left_contact & right_contact
            has_stable_simultaneous = (
                simultaneous.numel() >= stable_frames
                and bool(
                    simultaneous.unfold(0, stable_frames, 1)
                    .all(dim=-1)
                    .any()
                    .item()
                )
            )
            motion_requires_simultaneous_grasp.append(
                has_stable_simultaneous
            )
            loaded_dict['body_rot'] = loaded_dict['hoi_data'][:, 386:386+52*4].clone()

            body_rot = loaded_dict['body_rot'].view(-1, 52, 4)
            body_rot_vel = torch_utils.quat_sequence_angular_velocity(
                body_rot, self.fps_data
            )
            loaded_dict['body_rot_vel'] = self._clip_reference_angular_velocity(
                body_rot_vel
            ).view(-1, 52 * 3)

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(),
                                                    loaded_dict['root_rot'].clone(),
                                                    loaded_dict['dof_pos'].clone(),
                                                    loaded_dict['dof_vel'].clone(),
                                                    loaded_dict['body_pos'].clone(),
                                                    loaded_dict['body_rot'].clone(),
                                                    loaded_dict['body_pos_vel'].clone(),
                                                    loaded_dict['body_rot_vel'].clone(),
                                                    loaded_dict['obj1_pos'].clone(),
                                                    loaded_dict['obj1_rot'].clone(),
                                                    loaded_dict['obj1_pos_vel'].clone(),
                                                    loaded_dict['obj1_rot_vel'].clone(),
                                                    loaded_dict['obj2_pos'].clone(),
                                                    loaded_dict['obj2_rot'].clone(),
                                                    loaded_dict['obj2_pos_vel'].clone(),
                                                    loaded_dict['obj2_rot_vel'].clone(),
                                                    loaded_dict['ig1'].clone(),
                                                    loaded_dict['ig2'].clone(),
                                                    loaded_dict['contact_human'].clone(),
                                                    loaded_dict['contact_obj1'].clone(),
                                                    loaded_dict['contact_obj2'].clone(),
                                                    ), dim=-1)
            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(initk)]+[loaded_dict['hoi_data']], dim=0)
            hoi_datas.append(loaded_dict['hoi_data'])

            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(),
                                loaded_dict['root_rot'].clone(),
                                loaded_dict['root_pos_vel'].clone(),
                                loaded_dict['root_rot_vel'].clone(),
                                loaded_dict['dof_pos'].clone(),
                                loaded_dict['dof_vel'].clone(),
                                loaded_dict['obj1_pos'].clone(),
                                loaded_dict['obj1_rot'].clone(),
                                loaded_dict['obj1_pos_vel'].clone(),
                                loaded_dict['obj1_rot_vel'].clone(),
                                loaded_dict['obj2_pos'].clone(),
                                loaded_dict['obj2_rot'].clone(),
                                loaded_dict['obj2_pos_vel'].clone(),
                                loaded_dict['obj2_rot_vel'].clone(),
                                ), dim=-1)
            hoi_refs.append(hoi_ref)
        max_length = max(max_episode_length) + initk
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(max_episode_length, dtype=torch.long, device=self.device) + initk
        self._motion_requires_simultaneous_grasp = torch.tensor(
            motion_requires_simultaneous_grasp,
            device=self.device,
            dtype=torch.bool,
        )
        if self.cfg['env'].get('fullSequenceRollout', False):
            if self._state_init != InterMimic.StateInit.Start:
                raise ValueError(
                    "fullSequenceRollout requires stateInit: Start"
                )
            self.rollout_length = max_length
            print(
                f"[INFO] full-sequence rollout enabled: "
                f"rolloutLength={self.rollout_length}"
            )
        hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        # Stack on CPU, then move to self.device
        hoi_data = torch.stack(hoi_data, dim=0).to(self.device)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, topk, 1, 1).to(self.device)

        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2]), device=self.device)
        self.ref_reward[:, 0, :] = 1.0

        self.ref_index = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)

        # Evaluation metrics tracking per sequence (only if evaluation is enabled)
        if self.enable_evaluation:
            # Track visit counts for balanced sampling
            self._sequence_visit_count = torch.zeros([self.num_motions], device=self.hoi_refs.device, dtype=torch.long)
            self._eval_episode_count = torch.zeros((), device=self.device, dtype=torch.long)
            self._eval_completion_count = torch.zeros((), device=self.device, dtype=torch.long)
            self._eval_reach_count = torch.zeros((), device=self.device, dtype=torch.long)
            self._eval_correct_contact_count = torch.zeros((), device=self.device, dtype=torch.long)
            self._eval_stable_grasp_count = torch.zeros((), device=self.device, dtype=torch.long)
            self._eval_semantic_success_count = torch.zeros((), device=self.device, dtype=torch.long)
            self._eval_wrong_contact_steps = torch.zeros((), device=self.device, dtype=torch.long)
            self._eval_episode_count_per_seq = torch.zeros(
                self.num_motions, device=self.device, dtype=torch.long
            )
            self._eval_completion_count_per_seq = torch.zeros_like(
                self._eval_episode_count_per_seq
            )
            self._eval_semantic_success_count_per_seq = torch.zeros_like(
                self._eval_episode_count_per_seq
            )
            self._eval_wrong_contact_steps_per_seq = torch.zeros_like(
                self._eval_episode_count_per_seq
            )
            self._eval_reach_seen = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
            self._eval_correct_seen = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
            self._eval_stable_seen = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
            self._eval_contact_streak = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
            self._eval_simultaneous_contact_streak = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._eval_simultaneous_stable_seen = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._eval_final_object_error = torch.full((self.num_envs,), 1e6, device=self.device)
            self._eval_final_object_pos_error = torch.full(
                (self.num_envs, 2), 1e6, device=self.device
            )
            self._eval_final_object_rot_error_deg = torch.full(
                (self.num_envs, 2), 1e6, device=self.device
            )
            self._eval_active_env = torch.ones(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._eval_error_steps = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._eval_human_error_sum = torch.zeros(
                self.num_envs, device=self.device
            )
            self._eval_object_error_sum = torch.zeros(
                self.num_envs, device=self.device
            )
            self._eval_wrong_contact_steps_env = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._eval_result_recorded = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._eval_result_sequence = torch.full(
                (self.num_envs,), -1, device=self.device, dtype=torch.long
            )
            self._eval_result_steps = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._eval_result_completed = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._eval_result_reached = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_contacted = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_stable = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_stable_required = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_semantic = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_human_error = torch.full(
                (self.num_envs,), float('nan'), device=self.device
            )
            self._eval_result_object_error = torch.full(
                (self.num_envs,), float('nan'), device=self.device
            )
            self._eval_result_wrong_contact_steps = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._eval_result_final_object_pos_error = torch.full(
                (self.num_envs, 2), float('nan'), device=self.device
            )
            self._eval_result_final_object_rot_error_deg = torch.full(
                (self.num_envs, 2), float('nan'), device=self.device
            )
            self._eval_result_human_termination = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_object_termination = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_ig_termination = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_wrist_termination = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_object_phase_termination = torch.zeros_like(
                self._eval_result_completed
            )
            self._eval_result_contact_phase_termination = torch.zeros_like(
                self._eval_result_completed
            )

        if not hasattr(self, 'data_component_order'):
            self.create_component_stat(loaded_dict)

        # Pre-compute per-frame RSI sampling weights + adaptive rollout length
        self._rsi_weights = []
        transition_pre = 50   # ~1.3s: covers full reach phase
        transition_post = 10  # ~0.5s: covers grasp + early hold
        transition_boost = 4.0
        latest_onset = 0
        ch_start = self.data_component_index[self.data_component_order.index('contact_human')]

        for idx_m in range(len(hoi_datas)):
            T_m = max_episode_length[idx_m]
            w = torch.ones(T_m + initk)
            ch = hoi_data[idx_m, :T_m + initk, ch_start:ch_start + 52]
            left_contact = (ch[:, 17:33] > 0.5).any(dim=-1)
            right_contact = (ch[:, 36:52] > 0.5).any(dim=-1)
            for contact_mask in [left_contact, right_contact]:
                onset = None
                for f in range(len(contact_mask)):
                    if contact_mask[f] and onset is None:
                        onset = f
                        break
                if onset is not None:
                    latest_onset = max(latest_onset, onset)
                    start = max(0, onset - transition_pre)
                    end = min(T_m + initk, onset + transition_post)
                    w[start:end] = transition_boost
            self._rsi_weights.append(w.to(self.device))

        # Adaptive rollout: must cover from frame 0 past the latest contact onset + margin
        adaptive_rollout = latest_onset + transition_post + 30  # onset + post + 30 frames into contact
        cfg_rollout = self.rollout_length
        if adaptive_rollout > cfg_rollout:
            self.rollout_length = adaptive_rollout
            print(f"[INFO] rolloutLength auto-adjusted: {cfg_rollout} -> {self.rollout_length} "
                  f"(latest contact onset at frame {latest_onset})")

        return hoi_data

    def create_component_stat(self, loaded_dict):
        self.data_component_order = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot', 'body_pos_vel', 'body_rot_vel',
            'obj1_pos', 'obj1_rot', 'obj1_pos_vel', 'obj1_rot_vel',
            'obj2_pos', 'obj2_rot', 'obj2_pos_vel', 'obj2_rot_vel',
            'ig1', 'ig2', 'contact_human', 'contact_obj1', 'contact_obj2',
        ]

        # Precompute the sizes for each component.
        data_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.data_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.data_component_index = [sum(data_component_sizes[:i]) for i in range(len(data_component_sizes) + 1)]

        self.ref_component_order = [
            'root_pos', 'root_rot', 'root_pos_vel', 'root_rot_vel', 'dof_pos', 'dof_vel',
            'obj1_pos', 'obj1_rot', 'obj1_pos_vel', 'obj1_rot_vel',
            'obj2_pos', 'obj2_rot', 'obj2_pos_vel', 'obj2_rot_vel',
        ]

        ref_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.ref_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.ref_component_index = [sum(ref_component_sizes[:i]) for i in range(len(ref_component_sizes) + 1)]

    def extract_ref_component(self, var_name, data_id, ref_index, t):
        index = self.ref_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.ref_component_index[index]
        end = self.ref_component_index[index+1]
        
        return self.hoi_refs[data_id, ref_index, t, start:end]


    def extract_data_component(self, var_name, ref=False, data_id=None, t=None, obs=None):
        index = self.data_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.data_component_index[index]
        end = self.data_component_index[index+1]
        
        if ref and data_id is not None and t is not None:
            return self.hoi_data[data_id, t, start:end]
        
        if obs is not None:
            return obs[..., start:end]

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._table_handles = []
        self._load_target_asset()
        self._has_table = any(
            info is not None
            for motion_info in self._motion_table_info
            for info in motion_info
        )

        # Size aggregates from the assets PhysX actually loaded.  The previous
        # fixed 200-shape estimate overflowed once for every environment.
        object_asset = {
            name: self._target_asset[idx]
            for idx, name in enumerate(self.object_name)
        }
        max_extra_bodies = 0
        max_extra_shapes = 0
        for obj1_name, obj2_name in self._motion_obj_pairs:
            pair_bodies = 0
            pair_shapes = 0
            for obj_name in (obj1_name, obj2_name):
                asset = object_asset[obj_name]
                pair_bodies += self.gym.get_asset_rigid_body_count(asset)
                pair_shapes += self.gym.get_asset_rigid_shape_count(asset)
                table_asset = self._table_assets.get(obj_name)
                if table_asset is not None:
                    pair_bodies += self.gym.get_asset_rigid_body_count(table_asset)
                    pair_shapes += self.gym.get_asset_rigid_shape_count(table_asset)
            max_extra_bodies = max(max_extra_bodies, pair_bodies)
            max_extra_shapes = max(max_extra_shapes, pair_shapes)
        self._extra_agg_bodies = max_extra_bodies
        self._extra_agg_shapes = max_extra_shapes
        print(
            f"[PHYSICS] aggregate extra capacity: "
            f"bodies={self._extra_agg_bodies}, shapes={self._extra_agg_shapes}"
        )

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return   

    def _load_target_asset(self): # smplx
        asset_root = resolve_data_path("assets", "objects")
        self._target_asset = []
        points_num = []
        self.object_points = []
        for i, object_name in enumerate(self.object_name):

            asset_file = object_name + ".urdf"
            obj_file = resolve_data_path("assets", "objects", "objects", object_name, object_name + ".obj")
            max_convex_hulls = 64
            if isinstance(self.object_density, dict):
                if self._require_object_density and object_name not in self.object_density:
                    raise ValueError(
                        f"Missing explicit objectDensity for {object_name}"
                    )
                density = float(
                    self.object_density.get(
                        object_name, self.object_density.get('default', 500.0)
                    )
                )
            else:
                density = float(self.object_density)
        
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01

            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000


            self._target_asset.append(self.gym.load_asset(self.sim, str(asset_root), asset_file, asset_options))
            print(f"[PHYSICS] {object_name}: requested density={density:.3f} kg/m^3")

            mesh_obj = trimesh.load(str(obj_file), force='mesh')
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            # Keep the mesh's native local coordinates.  Centering only the
            # point cloud makes IG/contact geometry disagree with the URDF mesh.
            object_points = to_torch(object_points, device=self.device)
            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(object_points)

        self.object_points = torch.stack(self.object_points, dim=0).to(self.device)

        self._table_assets = {}
        table_thickness = 0.02
        for obj_name in self.object_name:
            info = self._table_info.get(obj_name)
            if info is None:
                continue
            opts = gymapi.AssetOptions()
            opts.fix_base_link = True
            self._table_assets[obj_name] = self.gym.create_box(
                self.sim, info['half_x'] * 2, info['half_y'] * 2, table_thickness, opts
            )
        self._table_thickness = table_thickness
        return

    def _create_object_actor(self, env_ptr, env_id, obj_name, col_group, col_filter, segmentation_id):
        """Create one object actor with physics properties. Table is created separately."""
        default_pose = gymapi.Transform()
        obj_idx = self.object_name.index(obj_name)
        handle = self.gym.create_actor(
            env_ptr, self._target_asset[obj_idx], default_pose,
            obj_name, col_group, col_filter, segmentation_id,
        )
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        for p in props:
            p.restitution = 0.0
            p.friction = 1.0
            p.rolling_friction = 0.1
            p.torsion_friction = 0.1
            p.contact_offset = 0.001
            p.rest_offset = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)
        self.gym.set_actor_scale(env_ptr, handle, self.ball_size)
        if env_id == 0:
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
            total_mass = sum(float(p.mass) for p in body_props)
            print(f"[PHYSICS] {obj_name}: Isaac actor mass={total_mass:.6f} kg")
        return handle

    def _create_table_actor(
        self, env_ptr, env_id, motion_id, object_slot, obj_name,
        col_group, segmentation_id,
    ):
        """Create table actor for the given object (after all object actors are created)."""
        info = self._motion_table_info[motion_id][object_slot]
        if info is None or obj_name not in self._table_assets:
            return None
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(info['init_x'], info['init_y'],
                                   info['table_top_z'] - self._table_thickness / 2)
        table_h = self.gym.create_actor(
            env_ptr, self._table_assets[obj_name], table_pose,
            f"table_{obj_name}", col_group, 1, segmentation_id,
        )
        self.gym.set_rigid_body_color(env_ptr, table_h, 0, gymapi.MESH_VISUAL,
                                      gymapi.Vec3(0.4, 0.3, 0.2))
        return table_h

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 1 if self.play_dataset else 0
        seg_id = 0

        motion_id = env_id % len(self._motion_obj_pairs)
        pair = self._motion_obj_pairs[motion_id]
        # Create all object actors FIRST so they occupy consecutive actor indices
        h1 = self._create_object_actor(env_ptr, env_id, pair[0], col_group, col_filter, seg_id)
        h2 = self._create_object_actor(env_ptr, env_id, pair[1], col_group, col_filter, seg_id)
        self._target_handles.append((h1, h2))
        # Create tables after both objects — tables are fixed and don't need state tensors
        table1 = self._create_table_actor(
            env_ptr, env_id, motion_id, 0, pair[0], col_group, seg_id
        )
        table2 = self._create_table_actor(
            env_ptr, env_id, motion_id, 1, pair[1], col_group, seg_id
        )
        self._table_handles.append((table1, table2))

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        all_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])
        self._target_states_1 = all_states[..., 1, :]
        self._target_states_2 = all_states[..., 2, :]

        base = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32)
        self._tar_actor_ids_1 = base + 1
        self._tar_actor_ids_2 = base + 2

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        cf = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)
        self._tar_contact_forces_1 = cf[..., self.num_bodies, :]
        self._tar_contact_forces_2 = cf[..., self.num_bodies + 1, :]
        if self.enable_evaluation:
            self._eval_exact_contact_enabled = bool(
                self.cfg['env'].get('useExactContactEvaluation', True)
            )
            if self._eval_exact_contact_enabled and self._use_gpu_pipeline:
                raise RuntimeError(
                    "Exact actor-pair contact evaluation is unavailable in "
                    "Isaac Gym's GPU pipeline. Re-run with --pipeline cpu, or "
                    "disable useExactContactEvaluation for explicitly labeled "
                    "proxy metrics."
                )
            self._eval_contact_debug_frames = set()
            self._eval_exact_any_record_seen = False
            self._eval_stable_frames = int(
                self.cfg['env'].get('evaluationStableFrames', 10)
            )
            self._eval_final_position_threshold = float(
                self.cfg['env'].get('evaluationFinalPositionThreshold', 0.05)
            )
            self._eval_final_rotation_threshold_deg = float(
                self.cfg['env'].get('evaluationFinalRotationThresholdDeg', 20.0)
            )
            self._eval_max_wrong_contact_steps = int(
                self.cfg['env'].get('evaluationMaxWrongContactSteps', 0)
            )
            self._eval_require_no_wrong_contact = bool(
                self.cfg['env'].get(
                    'evaluationRequireNoWrongContact', False
                )
            )
            self._eval_obj_body_indices = []
            self._eval_table_body_indices = []
            self._eval_left_hand_body_indices = []
            self._eval_right_hand_body_indices = []
            for env_ptr, (obj1, obj2), (table1, table2) in zip(
                self.envs, self._target_handles, self._table_handles
            ):
                self._eval_obj_body_indices.append((
                    self.gym.get_actor_rigid_body_index(
                        env_ptr, obj1, 0, gymapi.DOMAIN_ENV
                    ),
                    self.gym.get_actor_rigid_body_index(
                        env_ptr, obj2, 0, gymapi.DOMAIN_ENV
                    ),
                ))
                self._eval_table_body_indices.append((
                    self.gym.get_actor_rigid_body_index(
                        env_ptr, table1, 0, gymapi.DOMAIN_ENV
                    ) if table1 is not None else -2,
                    self.gym.get_actor_rigid_body_index(
                        env_ptr, table2, 0, gymapi.DOMAIN_ENV
                    ) if table2 is not None else -2,
                ))
                humanoid = self.humanoid_handles[len(self._eval_left_hand_body_indices)]
                self._eval_left_hand_body_indices.append(tuple(
                    self.gym.get_actor_rigid_body_index(
                        env_ptr, humanoid, body_id, gymapi.DOMAIN_ENV
                    )
                    for body_id in range(17, 33)
                ))
                self._eval_right_hand_body_indices.append(tuple(
                    self.gym.get_actor_rigid_body_index(
                        env_ptr, humanoid, body_id, gymapi.DOMAIN_ENV
                    )
                    for body_id in range(36, 52)
                ))
        return
    
    def _reset_target(self, env_ids):
        d, r, t = self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids]
        for st, prefix in [(self._target_states_1, 'obj1'), (self._target_states_2, 'obj2')]:
            st[env_ids, :3] = self.extract_ref_component(f'{prefix}_pos', d, r, t)
            st[env_ids, 3:7] = self.extract_ref_component(f'{prefix}_rot', d, r, t)
            if self.init_vel:
                st[env_ids, 7:10] = self.extract_ref_component(f'{prefix}_pos_vel', d, r, t)
                st[env_ids, 10:13] = self.extract_ref_component(f'{prefix}_rot_vel', d, r, t)
            else:
                st[env_ids, 7:13] = 0

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        ids = torch.cat([self._tar_actor_ids_1[env_ids], self._tar_actor_ids_2[env_ids]])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(ids), len(ids),
        )

    def _reset_envs(self, env_ids):
        # The player keeps done indices on the RL device.  Exact-contact
        # evaluation intentionally runs task tensors on CPU while inference
        # stays on CUDA, so normalize indices at this boundary.
        env_ids = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        if len(env_ids) > 0:
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()

            # Rebuild current HOI before constructing policy observations.
            # Otherwise the first action after RSI sees zeros or the previous
            # episode's interaction graph/finger state.
            self._compute_hoi_observations(env_ids)
            self._hist_obs[env_ids] = self._curr_obs[env_ids]
            self._curr_ref_obs[env_ids] = self.hoi_data[
                self.data_id[env_ids], self.progress_buf[env_ids]
            ]
            self._hist_ref_obs[env_ids] = self._curr_ref_obs[env_ids]
            self._compute_observations(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == InterMimic.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == InterMimic.StateInit.Start
              or self._state_init == InterMimic.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == InterMimic.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        self._reset_target(env_ids)
        if self.enable_evaluation:
            # Only the initial episode of each environment belongs to the
            # strict evaluation cohort.  Environments that fail early may be
            # reset by the vector wrapper while the remaining cohort runs.
            episode_ids = env_ids[self._eval_active_env[env_ids]]
            self._eval_reach_seen[episode_ids] = False
            self._eval_correct_seen[episode_ids] = False
            self._eval_stable_seen[episode_ids] = False
            self._eval_contact_streak[episode_ids] = 0
            self._eval_simultaneous_contact_streak[episode_ids] = 0
            self._eval_simultaneous_stable_seen[episode_ids] = False
            self._eval_final_object_error[episode_ids] = 1e6
            self._eval_final_object_pos_error[episode_ids] = 1e6
            self._eval_final_object_rot_error_deg[episode_ids] = 1e6
            self._eval_error_steps[episode_ids] = 0
            self._eval_human_error_sum[episode_ids] = 0
            self._eval_object_error_sum[episode_ids] = 0
            self._eval_wrong_contact_steps_env[episode_ids] = 0

        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        # During evaluation, prioritize undersampled sequences for balanced coverage
        if self.enable_evaluation:
            i = []
            for env_idx in env_ids:
                valid_motions = self._valid_motions_for_env(env_idx)

                # Get visit counts for valid motions
                visit_counts = self._sequence_visit_count[valid_motions]

                # Sample with inverse probability (prioritize less visited sequences)
                # Add 1 to avoid division by zero
                inv_counts = 1.0 / (visit_counts.float() + 1.0)
                probs = inv_counts / inv_counts.sum()

                # Sample based on inverse visit counts
                sampled_idx = torch.multinomial(probs, 1).item()
                selected_motion = valid_motions[sampled_idx]
                i.append(selected_motion)

            i = to_torch(i, device=self.device, dtype=torch.long)

            # Update visit counts
            for motion_id in i:
                self._sequence_visit_count[motion_id] += 1
        else:
            i = torch.stack([
                self._sample_motion_for_env(env_id)
                for env_id in env_ids
            ])

        if (self._state_init == InterMimic.StateInit.Random
            or self._state_init == InterMimic.StateInit.Hybrid):
            motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-self.rollout_length), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        elif (self._state_init == InterMimic.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)#.int()

        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self._hand_fail_counter[env_ids] = 0
        self._obj_fail_counter[env_ids] = 0
        self._contact_fail_counter[env_ids] = 0
        self._hand_fail_reset[env_ids] = False
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )

        return

    def cal_cdf(self, i, e):
        rewards = self.ref_reward[i[e], :, :max(1, self.max_episode_length[i[e]]-self.rollout_length)].clone() 
        ref_reward_sum = 1 / (rewards.sum(dim=0)) 
        prob = ref_reward_sum / ref_reward_sum.sum()
        cdf = torch.cumsum(prob, 0)
        return cdf

    def _sample_weighted_time(self, motion_idx):
        """Sample a start time with transition-window boosted weights."""
        max_t = max(1, self.max_episode_length[motion_idx].item() - self.rollout_length)
        w = self._rsi_weights[motion_idx][:max_t]
        if w.sum() < 1e-6:
            return torch.zeros(1, device=self.device, dtype=torch.long)
        prob = w / w.sum()
        return torch.multinomial(prob, 1)

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = torch.stack([
            self._sample_motion_for_env(env_id)
            for env_id in env_ids
        ])
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        motion_times = torch.cat([
            self._sample_weighted_time(i[e]) if env_ids[e] not in ref_reset_ids
            else torch.zeros((1,), device=self.device, dtype=torch.long)
            for e in range(num_envs)
        ])
        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self._hand_fail_counter[env_ids] = 0
        self._obj_fail_counter[env_ids] = 0
        self._contact_fail_counter[env_ids] = 0
        self._hand_fail_reset[env_ids] = False
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        if self.init_vel:
            self._humanoid_root_states[env_ids, 7:10] = root_vel
            self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        else:
            self._humanoid_root_states[env_ids, 7:13] = 0
        
        self._dof_pos[env_ids] = dof_pos
        if self.init_vel:
            self._dof_vel[env_ids] = dof_vel
        else:
            self._dof_vel[env_ids] = 0
        return

    def _compute_task_obs(self, env_ids=None, ref_obs=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            ts1, ts2 = self._target_states_1, self._target_states_2
        else:
            root_states = self._humanoid_root_states[env_ids]
            ts1, ts2 = self._target_states_1[env_ids], self._target_states_2[env_ids]

        obs1 = self.compute_obj_observations(root_states, ts1, ref_obs, obj_prefix='obj1')
        obs2 = self.compute_obj_observations(root_states, ts2, ref_obs, obj_prefix='obj2')
        return torch.cat([obs1, obs2], dim=-1)

    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids):
        # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        len_keypos = len(key_body_ids)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
        flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1], 
                                                heading_rot_expand_no_hand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
        flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(heading_inv_rot_expand_no_hand.shape[0] * heading_inv_rot_expand_no_hand.shape[1], 
                                                heading_inv_rot_expand_no_hand.shape[2])
        
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_pos = body_pos[:, key_body_ids, :]

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
        
        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
    
        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs)
        ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1) 
        body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_vel = body_vel[:, key_body_ids, :]
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        ref_body_ang_vel_no_hand = torch.cat((ref_body_ang_vel[:, :18*3], ref_body_ang_vel[:, 33*3:37*3]), dim=-1)
        body_ang_vel_no_hand = torch.cat((body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1)
        diff_global_ang_vel = ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)).view(-1, 22 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids]
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

        obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
        return obs
    
    def compute_obj_observations(self, root_states, tar_states, ref_obs, obj_prefix='obj1'):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = tar_states[:, 0:3]
        tar_rot = tar_states[:, 3:7]
        tar_vel = tar_states[:, 7:10]
        tar_ang_vel = tar_states[:, 10:13]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        local_tar_vel = quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

        _ref_obj_pos = self.extract_data_component(f'{obj_prefix}_pos', obs=ref_obs)
        diff_local_obj_pos = torch_utils.quat_rotate(heading_rot, _ref_obj_pos - tar_pos)

        ref_obj_rot = self.extract_data_component(f'{obj_prefix}_rot', obs=ref_obs)
        diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot = torch_utils.quat_to_tan_norm(
            torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot))

        ref_obj_vel = self.extract_data_component(f'{obj_prefix}_pos_vel', obs=ref_obs)
        diff_local_vel = torch_utils.quat_rotate(heading_rot, ref_obj_vel - tar_vel)

        ref_obj_ang_vel = self.extract_data_component(f'{obj_prefix}_rot_vel', obs=ref_obs)
        diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, ref_obj_ang_vel - tar_ang_vel)

        return torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos, diff_local_obj_rot, diff_local_vel, diff_local_ang_vel], dim=-1)
    
    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)

        ts = self.progress_buf[env_ids].clone()
        next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]] - 1)
        ref_obs = hoi_data[self.data_id[env_ids], next_ts].clone()
        obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
        task_obs = self._compute_task_obs(env_ids, ref_obs)
        obs = torch.cat([obs, task_obs], dim=-1)

        ig_all_1, ig_1, ref_ig_1 = self._compute_ig_obs_single(env_ids, ref_obs, 'ig1')
        ig_all_2, ig_2, ref_ig_2 = self._compute_ig_obs_single(env_ids, ref_obs, 'ig2')

        ref_dof = self.extract_data_component('dof_pos', obs=ref_obs)
        cur_dof = self.extract_data_component('dof_pos', obs=self._curr_obs[env_ids])
        diff_finger_dof = ref_dof[:, self._FINGER_DOF_IDX] - cur_dof[:, self._FINGER_DOF_IDX]

        return torch.cat((obs, ig_all_1, ig_all_2, ref_ig_1 - ig_1, ref_ig_2 - ig_2, diff_finger_dof), dim=-1)

    def _compute_ig_obs_single(self, env_ids, ref_obs, ig_key):
        ig = self.extract_data_component(ig_key, obs=self._curr_obs[env_ids]).view(env_ids.shape[0], -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig_key_bodies = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
        ig_all_flat = ig_all.view(env_ids.shape[0], -1)
        ref_ig = self.extract_data_component(ig_key, obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids, :]
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()
        ref_ig_flat = ref_ig.view(env_ids.shape[0], -1)
        return ig_all_flat, ig_key_bodies, ref_ig_flat
        
    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            self._curr_ref_obs[:] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            # Teacher policy always uses MLP (2 time steps: 1, 16)
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(self.hoi_data, None, 1), self._compute_observations_iter(self.hoi_data, None, 16)), dim=-1)

        else:
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            # Teacher policy always uses MLP (2 time steps: 1, 16)
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data, env_ids, 1), self._compute_observations_iter(self.hoi_data, env_ids, 16)), dim=-1)

        return
    
    def _compute_hoi_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._curr_obs[env_ids] = self.build_hoi_observations(
            self._rigid_body_pos[env_ids, 0, :], self._rigid_body_rot[env_ids, 0, :],
            self._rigid_body_vel[env_ids, 0, :], self._rigid_body_ang_vel[env_ids, 0, :],
            self._dof_pos[env_ids], self._dof_vel[env_ids], self._rigid_body_pos[env_ids],
            self._local_root_obs, self._root_height_obs, self._dof_obs_size,
            self._target_states_1[env_ids], self._target_states_2[env_ids],
            self._tar_contact_forces_1[env_ids], self._tar_contact_forces_2[env_ids],
            self._contact_forces[env_ids],
            self.object_points[self.obj1_id[self.data_id[env_ids]]],
            self.object_points[self.obj2_id[self.data_id[env_ids]]],
            self._rigid_body_rot[env_ids], self._rigid_body_vel[env_ids],
            self._rigid_body_ang_vel[env_ids],
        )

    def _compute_ig_for_object(self, body_pos, root_rot, target_states, object_points):
        """Compute SDF-based interaction graph for one object."""
        tar_pos = target_states[:, 0:3]
        tar_rot = target_states[:, 3:7]
        rot_ext = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        pts_ext = object_points.view(-1, 3)
        obj_pts = torch_utils.quat_rotate(rot_ext, pts_ext).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        ig = compute_sdf(body_pos, obj_pts).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_ext = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_ext, ig).view(tar_pos.shape[0], -1)
        return ig

    def build_hoi_observations(self, root_pos, root_rot, root_vel, root_ang_vel,
                               dof_pos, dof_vel, body_pos, local_root_obs, root_height_obs,
                               dof_obs_size, target_states_1, target_states_2,
                               target_contact_buf_1, target_contact_buf_2,
                               contact_buf, object_points_1, object_points_2,
                               body_rot, body_vel, body_rot_vel):

        contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
        tc1 = torch.any(torch.abs(target_contact_buf_1) > 0.1, dim=-1).float().unsqueeze(1)
        tc2 = torch.any(torch.abs(target_contact_buf_2) > 0.1, dim=-1).float().unsqueeze(1)

        ig1 = self._compute_ig_for_object(body_pos, root_rot, target_states_1, object_points_1)
        ig2 = self._compute_ig_for_object(body_pos, root_rot, target_states_2, object_points_2)

        obs = torch.cat((
            root_pos, root_rot, dof_pos, dof_vel,
            body_pos.reshape(body_pos.shape[0], -1),
            body_rot.reshape(body_rot.shape[0], -1),
            body_vel.reshape(body_vel.shape[0], -1),
            body_rot_vel.reshape(body_rot_vel.shape[0], -1),
            target_states_1, target_states_2,
            ig1, ig2, contact, tc1, tc2,
        ), dim=-1)
        return obs
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = self.compute_hoi_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                                           self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                                           self._enable_early_termination, self._termination_heights, self.start_times,
                                                                           self.rollout_length, self.kinematic_reset, self._hand_fail_reset
                                                                          )

        # Evaluation metrics update (assumes stateInit is "Start", so start_times is 0)
        if self.enable_evaluation:
            all_reset_id = torch.where(self.reset_buf)[0]
            # Record exactly one episode for each environment.  Inactive
            # environments can reset again while slower initial episodes run.
            reset_id = all_reset_id[self._eval_active_env[all_reset_id]]
            if reset_id.numel() > 0:
                completed = (
                    self.progress_buf[reset_id]
                    >= self.max_episode_length[self.data_id[reset_id]] - 1
                )
                reached = self._eval_reach_seen[reset_id].all(dim=-1)
                contacted = self._eval_correct_seen[reset_id].all(dim=-1)
                stable = self._eval_simultaneous_stable_seen[reset_id]
                seq_ids = self.data_id[reset_id]
                stable_required = (
                    self._motion_requires_simultaneous_grasp[seq_ids]
                )
                contact_requirement_met = torch.where(
                    stable_required, stable, contacted
                )
                final_position_ok = (
                    self._eval_final_object_pos_error[reset_id]
                    <= self._eval_final_position_threshold
                ).all(dim=-1)
                final_rotation_ok = (
                    self._eval_final_object_rot_error_deg[reset_id]
                    <= self._eval_final_rotation_threshold_deg
                ).all(dim=-1)
                wrong_contact_ok = (
                    self._eval_wrong_contact_steps_env[reset_id]
                    <= self._eval_max_wrong_contact_steps
                )
                if not self._eval_require_no_wrong_contact:
                    wrong_contact_ok = torch.ones_like(wrong_contact_ok)
                semantic = (
                    completed
                    & contact_requirement_met
                    & final_position_ok
                    & final_rotation_ok
                    & wrong_contact_ok
                )
                error_steps = self._eval_error_steps[reset_id].clamp(min=1)
                human_error = self._eval_human_error_sum[reset_id] / error_steps
                object_error = self._eval_object_error_sum[reset_id] / error_steps
                wrong_steps = self._eval_wrong_contact_steps_env[reset_id]

                self._eval_episode_count += reset_id.numel()
                self._eval_completion_count += completed.sum()
                self._eval_reach_count += reached.sum()
                self._eval_correct_contact_count += contacted.sum()
                self._eval_stable_grasp_count += stable.sum()
                self._eval_semantic_success_count += semantic.sum()
                self._eval_episode_count_per_seq.index_add_(
                    0, seq_ids, torch.ones_like(seq_ids)
                )
                self._eval_completion_count_per_seq.index_add_(
                    0, seq_ids, completed.long()
                )
                self._eval_semantic_success_count_per_seq.index_add_(
                    0, seq_ids, semantic.long()
                )
                self._eval_wrong_contact_steps += wrong_steps.sum()
                self._eval_wrong_contact_steps_per_seq.index_add_(
                    0, seq_ids, wrong_steps
                )

                self._eval_result_recorded[reset_id] = True
                self._eval_result_sequence[reset_id] = seq_ids
                self._eval_result_steps[reset_id] = self._eval_error_steps[reset_id]
                self._eval_result_completed[reset_id] = completed
                self._eval_result_reached[reset_id] = reached
                self._eval_result_contacted[reset_id] = contacted
                self._eval_result_stable[reset_id] = stable
                self._eval_result_stable_required[reset_id] = (
                    stable_required
                )
                self._eval_result_semantic[reset_id] = semantic
                self._eval_result_human_error[reset_id] = human_error
                self._eval_result_object_error[reset_id] = object_error
                self._eval_result_wrong_contact_steps[reset_id] = wrong_steps
                self._eval_result_final_object_pos_error[reset_id] = (
                    self._eval_final_object_pos_error[reset_id]
                )
                self._eval_result_final_object_rot_error_deg[reset_id] = (
                    self._eval_final_object_rot_error_deg[reset_id]
                )
                self._eval_result_human_termination[reset_id] = (
                    self._human_fail_reset[reset_id]
                )
                self._eval_result_object_termination[reset_id] = (
                    self._object_fail_reset[reset_id]
                )
                self._eval_result_ig_termination[reset_id] = (
                    self._ig_fail_reset[reset_id]
                )
                self._eval_result_wrist_termination[reset_id] = (
                    self._wrist_fail_reset[reset_id]
                    & self._enable_wrist_failure_termination
                )
                self._eval_result_object_phase_termination[reset_id] = (
                    self._object_phase_fail_reset[reset_id]
                    & self._enable_object_contact_phase_termination
                )
                self._eval_result_contact_phase_termination[reset_id] = (
                    self._contact_phase_fail_reset[reset_id]
                    & self._enable_contact_failure_termination
                )
                self._eval_active_env[reset_id] = False

        if self.reset_buf.sum() > 0 and self.psi > 1:
            reset_ind = (self.reset_buf == 1)
            data_id = self.data_id[reset_ind]
            max_episode_length = self.max_episode_length[data_id]
            if (max_episode_length < self.rollout_length).all():
                self._sum_reward[reset_ind] = 0
                return
            start_index, end_index = self.start_times[reset_ind], self.progress_buf[reset_ind]
            sum_reward = self._sum_reward[reset_ind].mean()
            if torch.rand(1)[0] < 0:
                self._sum_reward[reset_ind] = 0
                return
            self._sum_reward[reset_ind] = 0
            reset_ind = torch.logical_and(reset_ind, self.max_episode_length[self.data_id] > self.rollout_length)
            if reset_ind.sum() < 0.995:
                return
            curr_reward = self._curr_reward[reset_ind]
            state = self._curr_state[reset_ind]
            # Initialize the reward tensor with zeros
            reward = torch.zeros((curr_reward.shape[0], self.hoi_refs.shape[0], self.hoi_refs.shape[2]), device=curr_reward.device)
            end_i = torch.minimum(max_episode_length, self.rollout_length + start_index)

            assert (end_index < end_i).all()
            # Loop through each example in the batch to assign the values from curr_reward to the correct slices in reward

            # data_num, sample_choice, time, feature

            for i in range(curr_reward.shape[0]):
                if end_index[i] > start_index[i]+30:  # Ensure the indices are valid
                    index_tensor = torch.arange(start_index[i]+10, end_index[i]-10, device=start_index.device)
                    reward[i, data_id[i], start_index[i]+10:end_index[i]-10] = ((end_index[i] - index_tensor) / (end_i[i] - index_tensor))

            adjust_reward, adjust_reward_index = reward.max(dim=0)
            for i in range(reward.shape[1]):
                if self.max_episode_length[i] < self.rollout_length:
                    continue
                for j in range(reward.shape[2]):
                    if self.max_episode_length[i] - j < self.rollout_length:
                        break
                    value, index = self.ref_reward[i, 1:, j].min(dim=0)
                    index = index + 1
                    id1 = adjust_reward_index[i, j]
                    idx = j - start_index[adjust_reward_index[i, j]]

                    if idx > 0 and idx < self.rollout_length and adjust_reward[i, j] > 0.5:
                        self.ref_reward[i, index, j] = adjust_reward[i, j]
                        self.hoi_refs[i, index, j] = state[id1, idx]
            self.ref_reward[:, 1:, :] = self.ref_reward[:, 1:, :] * (1 - 1e-5)
        return

    def compute_hoi_reset(self, reset_buf, progress_buf, obs_buf, rigid_body_pos,
                          max_episode_length, enable_early_termination, termination_heights, 
                          start_times, rollout_length, reset_ig, contact_reset):

        reset, terminated = self.compute_humanoid_reset(reset_buf, progress_buf, obs_buf, rigid_body_pos,
                                                        max_episode_length, enable_early_termination, termination_heights, 
                                                        start_times, rollout_length)

        reset_ig *= (progress_buf > 1 + start_times)
        contact_reset *= (progress_buf > 1 + start_times)
                
        terminated = torch.where(torch.logical_or(reset_ig, contact_reset), torch.ones_like(reset_buf), terminated)
        reset = torch.where(reset.bool(), torch.ones_like(reset_buf), terminated)

        return reset, terminated

    _FINGER_DOF_IDX = list(range(51, 96)) + list(range(108, 153))
    _FINGER_BODY_IDS = list(range(18, 33)) + list(range(37, 52))

    def _init_residual_scale(self):
        self._residual_scale_per_dof = torch.full((153,), 0.3, device=self.device)
        self._residual_scale_per_dof[self._FINGER_DOF_IDX] = 0.6
        self._residual_scale_per_dof[self._wrist_dof_idx] = 0.5

        # Bound residuals in physical radians independently of broad XML
        # ranges (and of the base class's knee scale=5 special case).
        env_cfg = self.cfg['env']
        body_limit = float(env_cfg.get('residualBodyLimit', 0.30))
        wrist_limit = float(env_cfg.get('residualWristLimit', 0.40))
        finger_limit = float(env_cfg.get('residualFingerLimit', 0.45))
        self._residual_limit_per_dof = torch.full(
            (153,), body_limit, device=self.device
        )
        self._residual_limit_per_dof[self._wrist_dof_idx] = wrist_limit
        self._residual_limit_per_dof[self._FINGER_DOF_IDX] = finger_limit

    def _action_to_pd_targets(self, action):
        """Apply a bounded residual to the next reference frame."""
        if not hasattr(self, '_residual_scale_per_dof'):
            self._init_residual_scale()
        next_t = torch.minimum(
            self.progress_buf + 1,
            self.max_episode_length[self.data_id] - 1,
        )
        ref_dof = self.extract_data_component(
            'dof_pos', ref=True, data_id=self.data_id, t=next_t
        )
        residual = self._residual_scale_per_dof * self._pd_action_scale * action.clamp(-1.0, 1.0)
        residual = torch.maximum(
            torch.minimum(residual, self._residual_limit_per_dof),
            -self._residual_limit_per_dof,
        )
        target = ref_dof + residual
        return torch.maximum(
            torch.minimum(target, self.dof_limits_upper),
            self.dof_limits_lower,
        )

    def _compute_reward(self, actions):
        rb, human_reset, key_pos, ref_key_pos = self.compute_humanoid_reward(self.reward_weights)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward(self.reward_weights)
        rig, ig_reset = self.compute_ig_reward(self.reward_weights, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset = self.compute_cg_reward(self.reward_weights)

        # --- Finger DOF tracking ---
        ref_dof = self.extract_data_component('dof_pos', obs=self._curr_ref_obs)
        cur_dof = self.extract_data_component('dof_pos', obs=self._curr_obs)
        finger_err = ((ref_dof[:, self._FINGER_DOF_IDX] - cur_dof[:, self._FINGER_DOF_IDX]) ** 2).mean(dim=-1)
        self._r_finger = torch.exp(-self.reward_weights.get('finger', 5.0) * finger_err)

        # --- Contact phase detection (per-hand) ---
        ref_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        left_any = (ref_contact[:, 17:33] > 0.1).any(dim=-1).float()
        right_any = (ref_contact[:, 36:52] > 0.1).any(dim=-1).float()
        contact_phase = ((left_any + right_any) > 0).float()

        # --- Wrist error + reset (tighter: 15cm / 20 frames) ---
        sim_body_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(-1, 52, 3)
        ref_body_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(-1, 52, 3)
        left_wrist_err = (sim_body_pos[:, 17] - ref_body_pos[:, 17]).norm(dim=-1)
        right_wrist_err = (sim_body_pos[:, 36] - ref_body_pos[:, 36]).norm(dim=-1)
        hand_fail = (torch.max(left_wrist_err, right_wrist_err) > 0.15).float()
        self._hand_fail_counter = (self._hand_fail_counter + hand_fail) * hand_fail
        wrist_fail_reset = (
            (self._hand_fail_counter > 20)
            & (self.progress_buf > self.start_times + 10)
        )
        self._wrist_fail_reset = wrist_fail_reset
        self._hand_fail_reset = (
            wrist_fail_reset
            if self._enable_wrist_failure_termination
            else torch.zeros_like(wrist_fail_reset)
        )

        # --- Object trajectory reset (contact phase: obj off by >30% of its size for 20 frames) ---
        obj1_pos = self.extract_data_component('obj1_pos', obs=self._curr_obs)
        ref_obj1_pos = self.extract_data_component('obj1_pos', obs=self._curr_ref_obs)
        obj2_pos = self.extract_data_component('obj2_pos', obs=self._curr_obs)
        ref_obj2_pos = self.extract_data_component('obj2_pos', obs=self._curr_ref_obs)
        obj1_rot = self.extract_data_component('obj1_rot', obs=self._curr_obs)
        ref_obj1_rot = self.extract_data_component('obj1_rot', obs=self._curr_ref_obs)
        obj2_rot = self.extract_data_component('obj2_rot', obs=self._curr_obs)
        ref_obj2_rot = self.extract_data_component('obj2_rot', obs=self._curr_ref_obs)
        obj_thresholds = self._motion_obj_reset_thresholds[self.data_id]
        obj1_thresh = obj_thresholds[:, 0]
        obj2_thresh = obj_thresholds[:, 1]
        obj_fail = torch.max(
            left_any * ((obj1_pos - ref_obj1_pos).norm(dim=-1) > obj1_thresh).float(),
            right_any * ((obj2_pos - ref_obj2_pos).norm(dim=-1) > obj2_thresh).float(),
        )
        self._obj_fail_counter = (self._obj_fail_counter + obj_fail) * obj_fail
        obj_fail_reset = (self._obj_fail_counter > 20) & (self.progress_buf > self.start_times + 10)
        self._object_phase_fail_reset = obj_fail_reset
        if self._enable_object_contact_phase_termination:
            self._hand_fail_reset = self._hand_fail_reset | obj_fail_reset

        # --- Additive wrist tracking bonus (broad gradient, strong at 5-15cm) ---
        wrist_bonus = left_any * torch.exp(-5.0 * left_wrist_err) \
                    + right_any * torch.exp(-5.0 * right_wrist_err)

        # --- Grasp success bonus: correct hand-object pair only ---
        correct_left = self._correct_left_contact
        correct_right = self._correct_right_contact
        grasp_bonus = left_any * correct_left + right_any * correct_right
        wrong_contact = left_any * (self._left_hand_force_any - correct_left).clamp(min=0.0) \
                      + right_any * (self._right_hand_force_any - correct_right).clamp(min=0.0)

        # Align the optimized objective with the release criterion.  The base
        # object reward is human-heading-relative and can remain high when an
        # object finishes with a large world-frame orientation error.  Ramp in
        # a small global pose bonus during the final reference window.
        obj1_pos_error = (obj1_pos - ref_obj1_pos).norm(dim=-1)
        obj2_pos_error = (obj2_pos - ref_obj2_pos).norm(dim=-1)
        obj1_rot_diff = torch_utils.quat_mul_norm(
            torch_utils.quat_inverse(ref_obj1_rot), obj1_rot
        )
        obj2_rot_diff = torch_utils.quat_mul_norm(
            torch_utils.quat_inverse(ref_obj2_rot), obj2_rot
        )
        obj1_rot_error, _ = torch_utils.quat_to_angle_axis(obj1_rot_diff)
        obj2_rot_error, _ = torch_utils.quat_to_angle_axis(obj2_rot_diff)
        terminal_frames = max(
            1, int(self.cfg['env'].get('terminalObjectPoseFrames', 30))
        )
        frames_remaining = (
            self.max_episode_length[self.data_id] - 1 - self.progress_buf
        ).float()
        terminal_ramp = (
            1.0 - frames_remaining / float(terminal_frames)
        ).clamp(min=0.0, max=1.0)
        terminal_object_pose_bonus = terminal_ramp * 0.25 * (
            torch.exp(-50.0 * obj1_pos_error.square())
            + torch.exp(-50.0 * obj2_pos_error.square())
            + torch.exp(-2.0 * obj1_rot_error.abs())
            + torch.exp(-2.0 * obj2_rot_error.abs())
        )

        self._contact_fail_counter = (
            self._contact_fail_counter + contact_reset
        ) * contact_reset
        contact_fail_reset = (
            (
                self._contact_fail_counter
                > self._contact_failure_grace_frames
            ).any(dim=-1)
            & (self.progress_buf > self.start_times + 10)
        )
        self._contact_phase_fail_reset = contact_fail_reset
        if self._enable_contact_failure_termination:
            self._hand_fail_reset = self._hand_fail_reset | contact_fail_reset

        # --- Total reward ---
        finger_weight = float(self.cfg['env'].get('fingerBonusWeight', 0.05))
        wrist_weight = float(self.cfg['env'].get('wristBonusWeight', 0.30))
        grasp_weight = float(self.cfg['env'].get('graspBonusWeight', 0.05))
        wrong_contact_weight = float(self.cfg['env'].get('wrongContactPenalty', 0.0))
        terminal_pose_weight = float(
            self.cfg['env'].get('terminalObjectPoseBonusWeight', 0.0)
        )
        base_reward = rb * ro * rig
        if self._contact_reward_mode == 'legacy_multiplicative':
            base_reward = base_reward * rcg
        if self._contact_reward_mode == 'none':
            grasp_weight = 0.0
            wrong_contact_weight = 0.0
        self.rew_buf[:] = base_reward \
                        + finger_weight * self._r_finger \
                        + wrist_weight * wrist_bonus \
                        + grasp_weight * grasp_bonus \
                        + terminal_pose_weight * terminal_object_pose_bonus \
                        - wrong_contact_weight * wrong_contact

        kinematic_reset = torch.logical_or(human_reset, object_reset)
        self._human_fail_reset = human_reset.bool()
        self._object_fail_reset = object_reset.bool()
        self._ig_fail_reset = ig_reset.bool()
        self.contact_reset = self._contact_fail_counter
        self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        if self.psi > 1:
            index = torch.arange(self._curr_reward.shape[0], device=self.device)
            buffer_t = self.progress_buf - self.start_times
            self._curr_reward[index, buffer_t] = self.rew_buf
            self._sum_reward[index] += self.rew_buf
            self._curr_state[index, buffer_t, :] = torch.cat([
                self._humanoid_root_states,
                self._dof_pos,
                self._dof_vel,
                self._target_states_1,
                self._target_states_2,
            ], dim=1)

        human_error = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1)
        pts1, pts2 = obj_points
        rpts1, rpts2 = ref_obj_points
        object_error = ((pts1 - rpts1).norm(dim=-1).mean(dim=-1) + (pts2 - rpts2).norm(dim=-1).mean(dim=-1)) * 0.5

        if self.enable_evaluation:
            eval_correct_left = correct_left
            eval_correct_right = correct_right
            eval_wrong_contact = wrong_contact > 0
            if self._eval_exact_contact_enabled:
                (
                    eval_correct_left,
                    eval_correct_right,
                    _,
                    _,
                ) = self._compute_exact_pair_contact()
                eval_wrong_contact = self._eval_exact_wrong_contact

            ig1 = self.extract_data_component('ig1', obs=self._curr_obs).view(-1, 52, 3)
            ig2 = self.extract_data_component('ig2', obs=self._curr_obs).view(-1, 52, 3)
            reached_now = torch.stack(
                [ig1[:, 17].norm(dim=-1) < 0.10, ig2[:, 36].norm(dim=-1) < 0.10],
                dim=-1,
            )
            correct_now = torch.stack(
                [eval_correct_left > 0.5, eval_correct_right > 0.5], dim=-1
            )
            phase_now = torch.stack(
                [left_any > 0.5, right_any > 0.5], dim=-1
            )
            valid_contact = correct_now & phase_now
            active = self._eval_active_env
            self._eval_reach_seen[active] |= reached_now[active]
            self._eval_correct_seen[active] |= valid_contact[active]
            next_streak = (
                self._eval_contact_streak[active] + valid_contact[active].long()
            ) * valid_contact[active].long()
            self._eval_contact_streak[active] = next_streak
            self._eval_stable_seen[active] |= (
                next_streak >= self._eval_stable_frames
            )
            simultaneous = valid_contact.all(dim=-1)
            simultaneous_streak = (
                self._eval_simultaneous_contact_streak[active]
                + simultaneous[active].long()
            ) * simultaneous[active].long()
            self._eval_simultaneous_contact_streak[active] = simultaneous_streak
            self._eval_simultaneous_stable_seen[active] |= (
                simultaneous_streak >= self._eval_stable_frames
            )

            obj_pos_error = torch.stack([
                (self._target_states_1[:, :3] - ref_obj1_pos).norm(dim=-1),
                (self._target_states_2[:, :3] - ref_obj2_pos).norm(dim=-1),
            ], dim=-1)
            obj1_rot_diff = torch_utils.quat_mul_norm(
                torch_utils.quat_inverse(
                    self.extract_data_component('obj1_rot', obs=self._curr_ref_obs)
                ),
                self._target_states_1[:, 3:7],
            )
            obj2_rot_diff = torch_utils.quat_mul_norm(
                torch_utils.quat_inverse(
                    self.extract_data_component('obj2_rot', obs=self._curr_ref_obs)
                ),
                self._target_states_2[:, 3:7],
            )
            obj1_rot_angle, _ = torch_utils.quat_to_angle_axis(obj1_rot_diff)
            obj2_rot_angle, _ = torch_utils.quat_to_angle_axis(obj2_rot_diff)
            obj_rot_error_deg = torch.rad2deg(torch.stack(
                [obj1_rot_angle.abs(), obj2_rot_angle.abs()], dim=-1
            ))
            self._eval_final_object_error[active] = object_error[active]
            self._eval_final_object_pos_error[active] = obj_pos_error[active]
            self._eval_final_object_rot_error_deg[active] = obj_rot_error_deg[active]

            active_long = active.long()
            wrong_contact_step = eval_wrong_contact.long() * active_long
            self._eval_error_steps += active_long
            self._eval_human_error_sum += human_error * active.float()
            self._eval_object_error_sum += object_error * active.float()
            self._eval_wrong_contact_steps_env += wrong_contact_step

        self.extras['sub_rewards'] = {
            'rb': rb.mean().item(),
            'ro': ro.mean().item(),
            'ro1': self._ro1.mean().item(),
            'ro2': self._ro2.mean().item(),
            'rig': rig.mean().item(),
            'rcg': rcg.mean().item(),
            'r_finger': self._r_finger.mean().item(),
            'wrist_bonus': wrist_bonus.mean().item(),
            'grasp_bonus': grasp_bonus.mean().item(),
            'terminal_object_pose_bonus': terminal_object_pose_bonus.mean().item(),
            'wrong_contact': wrong_contact.mean().item(),
            'correct_left_contact': correct_left.mean().item(),
            'correct_right_contact': correct_right.mean().item(),
        }
        self.extras['errors'] = {
            'human_pose': human_error.mean().item(),
            'object_pose': object_error.mean().item(),
        }
        self.extras['reset_rates'] = {
            'human': human_reset.float().mean().item(),
            'object': object_reset.float().mean().item(),
            'ig': ig_reset.float().mean().item(),
            'contact': (contact_reset.sum(dim=-1) > 0).float().mean().item(),
            'contact_fail': contact_fail_reset.float().mean().item(),
            'contact_termination': (
                contact_fail_reset.float().mean().item()
                if self._enable_contact_failure_termination else 0.0
            ),
            'wrist_fail': wrist_fail_reset.float().mean().item(),
            'object_contact_phase_fail': obj_fail_reset.float().mean().item(),
            'hand_fail': self._hand_fail_reset.float().mean().item(),
        }

        # Per-step diagnostic logging (only for small envs, e.g. test mode)
        if self.num_envs <= 16 and hasattr(self, '_diag_file') and self._diag_file is not None:
            sim_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
            sim_left_any = (sim_contact[:, 17:33] > 0.1).any(dim=-1).float()

            for ei in range(self.num_envs):
                t = self.progress_buf[ei].item()
                st = self.start_times[ei].item()
                phase = "CONTACT" if contact_phase[ei] > 0.5 else "free"
                touch = "TOUCH" if sim_left_any[ei] > 0.5 else "no"
                obj1_pos = self.extract_data_component('obj1_pos', obs=self._curr_obs)
                left_wrist_obj = (sim_body_pos[ei, 17] - obj1_pos[ei]).norm().item()
                self._diag_file.write(
                    f"env{ei} t={t:3d} start={st:3d} phase={phase:7s} touch={touch:5s} "
                    f"rb={rb[ei]:.3f} ro={ro[ei]:.3f} "
                    f"rig={rig[ei]:.3f} rcg={rcg[ei]:.3f} "
                    f"r_fin={self._r_finger[ei]:.3f} rew={self.rew_buf[ei]:.3f} "
                    f"wrist_err={left_wrist_err[ei]:.4f} hand_obj={left_wrist_obj:.4f} "
                    f"hfail={self._hand_fail_counter[ei]:.0f} "
                    f"fin_err={finger_err[ei]:.4f}\n"
                )
                if self.reset_buf[ei] > 0:
                    self._diag_file.write(f"  >>> RESET env{ei} after {t - st} steps\n")
            self._diag_file.flush()

        return
    
    def compute_humanoid_reward(self, w):
        # body pos reward
        len_keypos = len(self._key_body_ids)
        key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._key_body_ids]
        
        ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._key_body_ids]
        
        ref_ig1 = self.extract_data_component('ig1', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig2 = self.extract_data_component('ig2', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = torch.min(ref_ig1.norm(dim=-1), ref_ig2.norm(dim=-1))
        weight_h = (-5 * ref_ig_norm).exp()
        weight_hp = weight_h.clone().detach()  
        ancle_toe_ids = [i for i in range(len_keypos) if 'Ankle' in self.key_bodies[i] or 'Toe' in self.key_bodies[i]]
        weight_hp[:, ancle_toe_ids] = 1

        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, self._key_body_ids],dim=-1)
        rp = torch.exp(-ep*w['p'])

        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        
        er = torch.mean(diff[:, :] * weight_hr, dim=-1)
        rr = torch.exp(-er*w['r'])
        
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        # body pos vel reward
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
        # epv = torch.mean(pos_vel ,dim=-1) # torch.zeros_like(ep)
        rpv = torch.exp(-epv*w['pv'])

        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        # body rot vel reward
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
        rrv = torch.exp(-erv*w['rv'])

        # energy penalty
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, 51*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()

        rb = rp*rr*rpv*rrv*energy
        human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5

        self.extras['human_sub'] = {
            'rp': rp.mean().item(),
            'rr': rr.mean().item(),
            'rpv': rpv.mean().item(),
            'rrv': rrv.mean().item(),
            'energy': energy.mean().item(),
        }
        return rb, human_reset, key_pos, ref_key_pos
    
    def _compute_single_obj_reward(self, w, obj_prefix, obj_id_per_motion):
        """Compute reward for a single object. Returns (ro, reset, obj_pts, ref_pts)."""
        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        root_rot = self.extract_data_component('root_rot', obs=self._curr_obs)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        obj_pos = self.extract_data_component(f'{obj_prefix}_pos', obs=self._curr_obs)
        obj_rot = self.extract_data_component(f'{obj_prefix}_rot', obs=self._curr_obs)
        local_obj_pos = obj_pos - root_pos
        local_obj_pos[..., -1] = obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)
        local_obj_rot = quat_mul(heading_rot, obj_rot)

        pts = self.object_points[obj_id_per_motion[self.data_id]]
        rot_ext = obj_rot.unsqueeze(1).repeat(1, pts.shape[1], 1).view(-1, 4)
        pts_ext = pts.view(-1, 3)
        obj_pts = torch_utils.quat_rotate(rot_ext, pts_ext).view(obj_rot.shape[0], pts.shape[1], 3) + obj_pos.unsqueeze(1)

        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_root_rot = self.extract_data_component('root_rot', obs=self._curr_ref_obs)
        ref_heading = torch_utils.calc_heading_quat_inv(ref_root_rot)

        ref_obj_pos = self.extract_data_component(f'{obj_prefix}_pos', obs=self._curr_ref_obs)
        ref_obj_rot = self.extract_data_component(f'{obj_prefix}_rot', obs=self._curr_ref_obs)
        ref_local_pos = ref_obj_pos - ref_root_pos
        ref_local_pos[..., -1] = ref_obj_pos[..., -1]
        ref_local_pos = quat_rotate(ref_heading, ref_local_pos)
        ref_local_rot = quat_mul(ref_heading, ref_obj_rot)

        ref_rot_ext = ref_obj_rot.unsqueeze(1).repeat(1, pts.shape[1], 1).view(-1, 4)
        ref_pts = torch_utils.quat_rotate(ref_rot_ext, pts_ext).view(obj_rot.shape[0], pts.shape[1], 3) + ref_obj_pos.unsqueeze(1)

        eop = ((ref_local_pos - local_obj_pos) ** 2).mean(dim=-1)
        rop = torch.exp(-eop * w['op'])
        diff_q = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_rot), local_obj_rot)
        diff_angle, _ = torch_utils.quat_to_angle_axis(diff_q)
        eor = diff_angle.view(-1, 1).mean(dim=-1)
        ror = torch.exp(-eor * w['or'])

        obj_pv = self.extract_data_component(f'{obj_prefix}_pos_vel', obs=self._curr_obs)
        ref_pv = self.extract_data_component(f'{obj_prefix}_pos_vel', obs=self._curr_ref_obs)
        ropv = torch.exp(-((ref_pv - obj_pv) ** 2).mean(dim=-1) * w['opv'])
        obj_rv = self.extract_data_component(f'{obj_prefix}_rot_vel', obs=self._curr_obs)
        ref_rv = self.extract_data_component(f'{obj_prefix}_rot_vel', obs=self._curr_ref_obs)
        rorv = torch.exp(-((ref_rv - obj_rv) ** 2).mean(dim=-1) * w['orv'])

        mask = (self.progress_buf - self.start_times > 2).float().unsqueeze(-1)
        hist_pv = self.extract_data_component(f'{obj_prefix}_pos_vel', obs=self._hist_obs)
        acc = (obj_pv - hist_pv) * self.fps_data * mask
        hist_rv = self.extract_data_component(f'{obj_prefix}_rot_vel', obs=self._hist_obs)
        racc = ((obj_rv - hist_rv) * self.fps_data * mask).view(-1, 3)
        energy = acc.pow(2).mean(dim=-1).mul(-w['eg2']).exp() * racc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()

        ro = rop * ror * ropv * rorv * energy
        reset = (obj_pts - ref_pts).norm(dim=-1).mean(dim=-1) > 0.5
        return ro, reset, obj_pts, ref_pts, eop, eor

    def compute_obj_reward(self, w):
        ro1, rst1, pts1, rpts1, eop1, eor1 = self._compute_single_obj_reward(w, 'obj1', self.obj1_id)
        ro2, rst2, pts2, rpts2, eop2, eor2 = self._compute_single_obj_reward(w, 'obj2', self.obj2_id)

        self._ro1, self._ro2 = ro1, ro2
        ro = ro1 * ro2
        object_reset = torch.logical_or(rst1, rst2)

        self.extras['object_sub'] = {
            'ro1': ro1.mean().item(), 'ro2': ro2.mean().item(),
            'obj1_pos_err': eop1.mean().item(), 'obj2_pos_err': eop2.mean().item(),
            'obj1_rot_err': eor1.mean().item(), 'obj2_rot_err': eor2.mean().item(),
        }
        return ro, object_reset, (pts1, pts2), (rpts1, rpts2)
    
    def _compute_single_ig_reward(self, w, key_pos, ref_key_pos, obj_pts, ref_obj_pts):
        """IG reward for one object."""
        n_key = len(self._key_body_ids)
        ig = key_pos.view(-1, n_key, 3).unsqueeze(2) - obj_pts.unsqueeze(1)
        ref_ig = ref_key_pos.view(-1, n_key, 3).unsqueeze(2) - ref_obj_pts.unsqueeze(1)
        w1 = 1 / torch.clamp((ig ** 2).sum(dim=-1), min=0.01)
        w1 = w1 / w1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        w2 = 1 / torch.clamp((ref_ig ** 2).sum(dim=-1), min=0.01)
        w2 = w2 / w2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        eig = ((ig - ref_ig) ** 2).sum(dim=-1) * (w1 + w2)
        rig = torch.exp(-w['ig'] * eig.sum(dim=-1).sum(dim=-1) * 0.5)
        r1 = (((ig - ref_ig) ** 2).sum(-1).sqrt() / torch.clamp((ref_ig ** 2).sum(-1).sqrt(), min=0.5)).max(-1)[0].max(-1)[0] > 2
        r2 = (((ig - ref_ig) ** 2).sum(-1).sqrt() / torch.clamp((ig ** 2).sum(-1).sqrt(), min=0.5)).max(-1)[0].max(-1)[0] > 2
        return rig, torch.logical_or(r1, r2)

    def compute_ig_reward(self, w, key_pos, ref_key_pos, obj_points_pair, ref_obj_points_pair):
        pts1, pts2 = obj_points_pair
        rpts1, rpts2 = ref_obj_points_pair
        rig1, rst1 = self._compute_single_ig_reward(w, key_pos, ref_key_pos, pts1, rpts1)
        rig2, rst2 = self._compute_single_ig_reward(w, key_pos, ref_key_pos, pts2, rpts2)
        self.extras.setdefault('ig_sub', {}).update({
            'rig1': rig1.mean().item(), 'rig2': rig2.mean().item(),
        })
        return rig1 * rig2, torch.logical_or(rst1, rst2)
    
    def _compute_pair_contact(self):
        """Approximate intended hand-object contacts entirely on GPU.

        This signal remains the control reward/reset input in every mode so
        instrumentation cannot change the rollout being measured.  Strict
        evaluation separately calls ``_compute_exact_pair_contact``.
        """
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        ig1 = self.extract_data_component('ig1', obs=self._curr_obs).view(-1, 52, 3)
        ig2 = self.extract_data_component('ig2', obs=self._curr_obs).view(-1, 52, 3)

        left_force = human_contact[:, 17:33] > 0.1
        right_force = human_contact[:, 36:52] > 0.1
        left_near_obj1 = ig1[:, 17:33].norm(dim=-1) < self._correct_contact_distance
        right_near_obj2 = ig2[:, 36:52].norm(dim=-1) < self._correct_contact_distance

        correct_left = (left_force & left_near_obj1).any(dim=-1).float()
        correct_right = (right_force & right_near_obj2).any(dim=-1).float()
        left_force_any = left_force.any(dim=-1).float()
        right_force_any = right_force.any(dim=-1).float()
        return correct_left, correct_right, left_force_any, right_force_any

    def _compute_exact_pair_contact(self):
        """Classify PhysX contact records by environment-domain body index."""
        # Preview 4's simulation-wide API loses reliable environment mapping
        # with multiple environments.  Query each environment and attach the
        # known environment id ourselves.
        env_contacts = []
        empty_contacts = None
        for env_id, env_ptr in enumerate(self.envs):
            records = self.gym.get_env_rigid_contacts(env_ptr)
            if empty_contacts is None:
                empty_contacts = records
            if records.size == 0:
                continue
            records = records.copy()
            records['env0'] = np.where(records['body0'] >= 0, env_id, -1)
            records['env1'] = np.where(records['body1'] >= 0, env_id, -1)
            env_contacts.append(records)
        contacts = (
            np.concatenate(env_contacts)
            if env_contacts
            else empty_contacts
        )
        if contacts is None:
            raise RuntimeError("No Isaac Gym environments available for contact evaluation")
        required_fields = {'body0', 'body1', 'env0', 'env1', 'lambda'}
        if contacts.dtype.names is None or not required_fields.issubset(contacts.dtype.names):
            raise RuntimeError(
                "Isaac Gym returned an unsupported rigid-contact schema; "
                "strict evaluation refuses to fall back to a distance proxy"
            )

        correct_left_np = np.zeros(self.num_envs, dtype=np.bool_)
        correct_right_np = np.zeros(self.num_envs, dtype=np.bool_)
        left_any_np = np.zeros(self.num_envs, dtype=np.bool_)
        right_any_np = np.zeros(self.num_envs, dtype=np.bool_)
        wrong_np = np.zeros(self.num_envs, dtype=np.bool_)

        if contacts.size > 0:
            self._eval_exact_any_record_seen = True
            body0 = contacts['body0'].astype(np.int64, copy=False)
            body1 = contacts['body1'].astype(np.int64, copy=False)
            env0 = contacts['env0'].astype(np.int64, copy=False)
            env1 = contacts['env1'].astype(np.int64, copy=False)
            force = np.abs(contacts['lambda'])
            env_id = np.where(env0 >= 0, env0, env1)
            valid = (
                (force > 1e-8)
                & (env_id >= 0)
                & (env_id < self.num_envs)
                & ((env0 < 0) | (env1 < 0) | (env0 == env1))
            )

            if os.environ.get('THEIA_DEBUG_CONTACTS') == '1':
                frame = int(self.progress_buf[0].item())
                debug_targets = {72, 92, 116, 140, 180, 240, 290}
                if (
                    frame in debug_targets
                    and frame not in self._eval_contact_debug_frames
                ):
                    self._eval_contact_debug_frames.add(frame)
                    debug_mask = valid & (env_id == 0)
                    debug_rows = contacts[debug_mask]
                    print(
                        f"[CONTACT_DEBUG] frame={frame} records={contacts.size} "
                        f"env0_valid={debug_rows.size} "
                        f"left={self._eval_left_hand_body_indices[0]} "
                        f"right={self._eval_right_hand_body_indices[0]} "
                        f"objects={self._eval_obj_body_indices[0]} "
                        f"tables={self._eval_table_body_indices[0]}"
                    )
                    for row in debug_rows[:40]:
                        print(
                            "[CONTACT_DEBUG] "
                            f"body0={int(row['body0'])} body1={int(row['body1'])} "
                            f"env0={int(row['env0'])} env1={int(row['env1'])} "
                            f"force={float(row['lambda']):.6g}"
                        )

            body0 = body0[valid]
            body1 = body1[valid]
            env_id = env_id[valid]
            obj_ids = np.asarray(self._eval_obj_body_indices, dtype=np.int64)
            table_ids = np.asarray(self._eval_table_body_indices, dtype=np.int64)

            # Actor layout is identical across environments, but retain the
            # per-environment lists so this remains correct if assets differ.
            for idx in np.unique(env_id):
                mask = env_id == idx
                b0 = body0[mask]
                b1 = body1[mask]
                left_ids = np.asarray(
                    self._eval_left_hand_body_indices[int(idx)], dtype=np.int64
                )
                right_ids = np.asarray(
                    self._eval_right_hand_body_indices[int(idx)], dtype=np.int64
                )
                left0 = np.isin(b0, left_ids)
                left1 = np.isin(b1, left_ids)
                right0 = np.isin(b0, right_ids)
                right1 = np.isin(b1, right_ids)
                left_any_np[idx] = np.any(left0 | left1)
                right_any_np[idx] = np.any(right0 | right1)

                obj1, obj2 = obj_ids[idx]
                table1, table2 = table_ids[idx]
                correct_left_np[idx] = np.any(
                    (left0 & (b1 == obj1)) | (left1 & (b0 == obj1))
                )
                correct_right_np[idx] = np.any(
                    (right0 & (b1 == obj2)) | (right1 & (b0 == obj2))
                )

                table0 = (b0 == table1) | (b0 == table2)
                table1_mask = (b1 == table1) | (b1 == table2)
                ground0 = b0 == -1
                ground1 = b1 == -1
                left_wrong = (
                    (left0 & ((b1 == obj2) | table1_mask | ground1))
                    | (left1 & ((b0 == obj2) | table0 | ground0))
                )
                right_wrong = (
                    (right0 & ((b1 == obj1) | table1_mask | ground1))
                    | (right1 & ((b0 == obj1) | table0 | ground0))
                )
                wrong_np[idx] = np.any(left_wrong | right_wrong)

        self._eval_exact_wrong_contact = torch.as_tensor(
            wrong_np, device=self.device
        )
        return tuple(
            torch.as_tensor(value, device=self.device, dtype=torch.float)
            for value in (
                correct_left_np, correct_right_np, left_any_np, right_any_np
            )
        )

    def compute_cg_reward(self, w):
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        left_contact_hand_ids = list(range(17, 33))
        
        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        correct_left, correct_right, left_force_any, right_force_any = self._compute_pair_contact()
        self._correct_left_contact = correct_left
        self._correct_right_contact = correct_right
        self._left_hand_force_any = left_force_any
        self._right_hand_force_any = right_force_any

        ecg_left = ref_left_contact_hand_any * (1.0 - correct_left)
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left*w['cg_hand'])) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)

        right_contact_hand_ids = list(range(36, 52))
        
        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()

        contact_reset = torch.cat([ 
                                (ref_left_contact_hand_any * (1.0 - correct_left)).unsqueeze(-1),
                                (ref_right_contact_hand_any * (1.0 - correct_right)).unsqueeze(-1),
                                ], dim=-1)
        
        ecg_right = ref_right_contact_hand_any * (1.0 - correct_right)
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right*w['cg_hand'])) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)
        
        rcg_hand = rcg_left * rcg_right
        # Foot-ground and object-support forces are necessary contacts.  Do not
        # reuse the hand-only {-1, 1} labels to penalize all body contacts.
        return rcg_hand, contact_reset
    
    def play_dataset_step(self, time):

        t = time
        if t == 0:
            self.data_id = torch.stack([
                self._sample_motion_for_env(env_id)
                for env_id in range(self.num_envs)
            ])
        env_ids = to_torch([i for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        t = to_torch(
                [
                    t if t < self.max_episode_length[self.data_id[i]] else self.max_episode_length[self.data_id[i]]-1
                    for i in range(self.num_envs)
                ],
                device=self.device,
                dtype=torch.long
            )
        ### update objects ###
        for st, prefix in [(self._target_states_1, 'obj1'), (self._target_states_2, 'obj2')]:
            st[env_ids, :3] = self.extract_data_component(f'{prefix}_pos', True, self.data_id[env_ids], t)
            st[env_ids, 3:7] = self.extract_data_component(f'{prefix}_rot', True, self.data_id[env_ids], t)
            st[env_ids, 7:13] = 0

        ### update subject ###   
        _humanoid_root_pos = self.extract_data_component('root_pos', True, self.data_id[env_ids], t)
        _humanoid_root_rot = self.extract_data_component('root_rot', True, self.data_id[env_ids], t)
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(self._humanoid_root_states[:, 7:10])
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(self._humanoid_root_states[:, 10:13])
        
        self._dof_pos[env_ids] = self.extract_data_component('dof_pos', True, self.data_id[env_ids], t)
        self._dof_vel[env_ids] = self.extract_data_component('dof_vel', True, self.data_id[env_ids], t)


        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        tar_ids = torch.cat([self._tar_actor_ids_1[env_ids], self._tar_actor_ids_2[env_ids]])
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(tar_ids), len(tar_ids))

        self.gym.simulate(self.sim)
        self._refresh_sim_tensors()

        obj1_contact = self.extract_data_component('contact_obj1', True, self.data_id[env_ids], t)
        obj1_contact = torch.any(obj1_contact > 0.1, dim=-1)
        human_contact = self.extract_data_component('contact_human', True, self.data_id[env_ids], t)
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                env_ptr = self.envs[env_id]
                h1, h2 = self._target_handles[env_id]
                for handle in [h1, h2]:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(0., 0., 1.))

                if obj1_contact[env_id] == True:
                    self.gym.set_rigid_body_color(env_ptr, h1, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(1., 0., 0.))
                    
                handle = self.humanoid_handles[env_id]
                for j in range(self.num_bodies):
                    if human_contact[env_id, j] > 0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(1., 0., 0.))
                    elif human_contact[env_id, j] > -0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 1., 0.))
                    else:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 0., 1.))
        self.render(t=t)

        return
    

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:
            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.progress_buf[env_ids]
                dataname = self.motion_file[-1][6:-3]
                images_dir = resolve_data_path("images", dataname, must_exist=False)
                images_dir.mkdir(parents=True, exist_ok=True)
                rgb_filename = images_dir / ("rgb_env%d_frame%05d.png" % (env_ids, frame_id))
                self.gym.write_viewer_image_to_file(self.viewer, str(rgb_filename))
        return

    def evaluation_complete(self):
        """Whether every environment's initial episode has been recorded."""
        return (
            self.enable_evaluation
            and bool((~self._eval_active_env).all().item())
        )

    @staticmethod
    def _wilson_interval(successes, total, z=1.959963984540054):
        if total <= 0:
            return [None, None]
        probability = successes / total
        z2 = z * z
        denominator = 1.0 + z2 / total
        centre = (probability + z2 / (2.0 * total)) / denominator
        radius = (
            z
            * math.sqrt(
                (probability * (1.0 - probability) + z2 / (4.0 * total))
                / total
            )
            / denominator
        )
        return [max(0.0, centre - radius), min(1.0, centre + radius)]

    @staticmethod
    def _error_statistics(values, failed_mask=None):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        failed_values = np.asarray([], dtype=np.float64)
        if failed_mask is not None:
            failed_mask = np.asarray(failed_mask, dtype=np.bool_)
            failed_values = values[finite & failed_mask]
        values = values[finite]
        if values.size == 0:
            return {
                'mean': None, 'median': None, 'p95': None,
                'failed_mean': None,
            }
        failed_mean = None
        if failed_values.size > 0:
            failed_mean = float(failed_values.mean())
        return {
            'mean': float(values.mean()),
            'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)),
            'failed_mean': failed_mean,
        }

    def print_final_eval_summary(self):
        """Print and persist strict, episode-level evaluation results."""
        if not self.enable_evaluation:
            return
        if (
            self._eval_exact_contact_enabled
            and not self._eval_exact_any_record_seen
        ):
            raise RuntimeError(
                "Exact contact evaluation received zero PhysX contact records "
                "for the entire run; refusing to report false zero-contact metrics"
            )

        recorded = self._eval_result_recorded.detach().cpu().numpy().astype(bool)
        recorded_envs = np.flatnonzero(recorded)
        actual_episodes = int(recorded.sum())
        expected_episodes = int(self.num_envs)

        result_sequence = self._eval_result_sequence.detach().cpu().numpy()
        result_steps = self._eval_result_steps.detach().cpu().numpy()
        result_completed = self._eval_result_completed.detach().cpu().numpy()
        result_reached = self._eval_result_reached.detach().cpu().numpy()
        result_contacted = self._eval_result_contacted.detach().cpu().numpy()
        result_stable = self._eval_result_stable.detach().cpu().numpy()
        result_stable_required = (
            self._eval_result_stable_required.detach().cpu().numpy()
        )
        result_semantic = self._eval_result_semantic.detach().cpu().numpy()
        result_human_error = self._eval_result_human_error.detach().cpu().numpy()
        result_object_error = self._eval_result_object_error.detach().cpu().numpy()
        result_wrong_steps = (
            self._eval_result_wrong_contact_steps.detach().cpu().numpy()
        )
        result_final_pos = (
            self._eval_result_final_object_pos_error.detach().cpu().numpy()
        )
        result_final_rot = (
            self._eval_result_final_object_rot_error_deg.detach().cpu().numpy()
        )
        termination_results = {
            'human_tracking': (
                self._eval_result_human_termination.detach().cpu().numpy()
            ),
            'object_tracking': (
                self._eval_result_object_termination.detach().cpu().numpy()
            ),
            'interaction_geometry': (
                self._eval_result_ig_termination.detach().cpu().numpy()
            ),
            'wrist_tracking': (
                self._eval_result_wrist_termination.detach().cpu().numpy()
            ),
            'object_contact_phase': (
                self._eval_result_object_phase_termination.detach().cpu().numpy()
            ),
            'required_contact_phase': (
                self._eval_result_contact_phase_termination.detach().cpu().numpy()
            ),
        }

        episode_rows = []
        eval_condition = os.environ.get('THEIA_EVAL_CONDITION', '')
        eval_training_seed = os.environ.get('THEIA_TRAINING_SEED', '')
        eval_seed = os.environ.get('THEIA_EVAL_SEED', '')
        eval_run_id = os.environ.get('THEIA_EVAL_RUN_ID', '')
        for env_id in recorded_envs.tolist():
            seq_id = int(result_sequence[env_id])
            episode_rows.append({
                'run_id': eval_run_id,
                'condition': eval_condition,
                'training_seed': eval_training_seed,
                'eval_seed': eval_seed,
                'env_id': env_id,
                # Environment i is deterministically bound to motion i % N.
                # Keeping this explicit lets the paper pipeline prove that
                # every reference received exactly K independent replicas.
                'trial_id': env_id // self.num_motions,
                'sequence_id': seq_id,
                'sequence': os.path.basename(self.motion_file[seq_id]),
                'reference_frames': int(
                    self.max_episode_length[seq_id].item()
                ),
                'fps': self.fps_data,
                'steps': int(result_steps[env_id]),
                'completed': int(result_completed[env_id]),
                'reached_both': int(result_reached[env_id]),
                'contacted_both': int(result_contacted[env_id]),
                'simultaneous_stable_grasp': int(result_stable[env_id]),
                'simultaneous_stable_grasp_required': int(
                    result_stable_required[env_id]
                ),
                'semantic_success': int(result_semantic[env_id]),
                'wrong_contact_steps': int(result_wrong_steps[env_id]),
                'mean_human_error_m': float(result_human_error[env_id]),
                'mean_object_surface_error_m': float(result_object_error[env_id]),
                'final_obj1_position_error_m': float(result_final_pos[env_id, 0]),
                'final_obj2_position_error_m': float(result_final_pos[env_id, 1]),
                'final_obj1_rotation_error_deg': float(result_final_rot[env_id, 0]),
                'final_obj2_rotation_error_deg': float(result_final_rot[env_id, 1]),
                **{
                    f'terminated_{name}': int(values[env_id])
                    for name, values in termination_results.items()
                },
            })

        def metric_summary(values):
            count = int(np.asarray(values)[recorded].sum())
            rate = count / actual_episodes if actual_episodes else 0.0
            return {
                'count': count,
                'rate': rate,
                'wilson_95': self._wilson_interval(count, actual_episodes),
            }

        failed = ~result_semantic[recorded]
        human_errors = result_human_error[recorded]
        object_errors = result_object_error[recorded]
        sequence_summaries = []
        termination_rows = []
        for seq_id, path in enumerate(self.motion_file):
            sequence_mask = recorded & (result_sequence == seq_id)
            sequence_episodes = int(sequence_mask.sum())
            sequence_successes = int(result_semantic[sequence_mask].sum())
            sequence_summaries.append({
                'sequence_id': seq_id,
                'sequence': os.path.basename(path),
                'episodes': sequence_episodes,
                'completed': int(result_completed[sequence_mask].sum()),
                'semantic_successes': sequence_successes,
                'semantic_rate': (
                    sequence_successes / sequence_episodes
                    if sequence_episodes else None
                ),
                'simultaneous_stable_grasp_required': bool(
                    self._motion_requires_simultaneous_grasp[
                        seq_id
                    ].item()
                ),
                'wrong_contact_steps': int(result_wrong_steps[sequence_mask].sum()),
            })
            termination_rows.append({
                'sequence_id': seq_id,
                'sequence': os.path.basename(path),
                'episodes': sequence_episodes,
                'completed': int(result_completed[sequence_mask].sum()),
                **{
                    name: int(values[sequence_mask].sum())
                    for name, values in termination_results.items()
                },
            })

        summary = {
            'schema_version': 2,
            'run': {
                'run_id': eval_run_id,
                'condition': eval_condition,
                'training_seed': eval_training_seed,
                'eval_seed': eval_seed,
                'fps': self.fps_data,
                'num_references': self.num_motions,
                'trials_per_reference': (
                    expected_episodes // self.num_motions
                    if expected_episodes % self.num_motions == 0 else None
                ),
            },
            'expected_episodes': expected_episodes,
            'actual_episodes': actual_episodes,
            'complete_cohort': actual_episodes == expected_episodes,
            'contact_source': (
                'physx_actor_pair'
                if self._eval_exact_contact_enabled
                else 'net_force_plus_distance_proxy'
            ),
            'thresholds': {
                'stable_frames': self._eval_stable_frames,
                'final_position_m': self._eval_final_position_threshold,
                'final_rotation_deg': self._eval_final_rotation_threshold_deg,
                'max_wrong_contact_steps': self._eval_max_wrong_contact_steps,
                'require_no_wrong_contact': (
                    self._eval_require_no_wrong_contact
                ),
            },
            'metrics': {
                'completion': metric_summary(result_completed),
                'reach_both': metric_summary(result_reached),
                'contact_both': metric_summary(result_contacted),
                'simultaneous_stable_grasp': metric_summary(result_stable),
                'semantic_success': metric_summary(result_semantic),
                'wrong_contact_steps': int(result_wrong_steps[recorded].sum()),
            },
            'errors': {
                'human_pose_m': self._error_statistics(human_errors, failed),
                'object_surface_m': self._error_statistics(object_errors, failed),
            },
            'sequences': sequence_summaries,
        }

        output_dir = os.environ.get('THEIA_EVAL_OUTPUT_DIR')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, 'episodes.csv')
            json_path = os.path.join(output_dir, 'summary.json')
            termination_path = os.path.join(
                output_dir, 'termination_causes.csv'
            )
            if episode_rows:
                with open(csv_path, 'w', newline='') as csv_file:
                    writer = csv.DictWriter(
                        csv_file, fieldnames=list(episode_rows[0].keys())
                    )
                    writer.writeheader()
                    writer.writerows(episode_rows)
            if termination_rows:
                with open(termination_path, 'w', newline='') as csv_file:
                    writer = csv.DictWriter(
                        csv_file, fieldnames=list(termination_rows[0].keys())
                    )
                    writer.writeheader()
                    writer.writerows(termination_rows)
            with open(json_path, 'w') as json_file:
                json.dump(summary, json_file, indent=2, sort_keys=True)
                json_file.write('\n')
            print(
                "  Machine-readable results: "
                f"{json_path}, {csv_path}, {termination_path}"
            )

        metrics = summary['metrics']
        errors = summary['errors']
        contact_label = (
            'Actor-pair Contact Both'
            if self._eval_exact_contact_enabled
            else 'Proxy Contact Both'
        )
        semantic_label = (
            'Semantic Success'
            if self._eval_exact_contact_enabled
            else 'Semantic Success (proxy contact)'
        )
        print("\n" + "=" * 72)
        print("STRICT EPISODE-LEVEL EVALUATION:")
        print(f"  Contact Source: {summary['contact_source']}")
        print(f"  Episodes: {actual_episodes}/{expected_episodes}")
        for label, key in [
            ('Completion', 'completion'),
            ('Reach Both Objects', 'reach_both'),
            (contact_label, 'contact_both'),
            ('Simultaneous Stable Grasp', 'simultaneous_stable_grasp'),
            (semantic_label, 'semantic_success'),
        ]:
            metric = metrics[key]
            low, high = metric['wilson_95']
            interval = (
                f", Wilson95={low:.2%}-{high:.2%}"
                if low is not None else ""
            )
            print(
                f"  {label}: {metric['rate']:.2%} "
                f"({metric['count']}/{actual_episodes}{interval})"
            )
        print(f"  Wrong-contact Steps: {metrics['wrong_contact_steps']}")
        def format_error(value):
            return f"{value:.4f}" if value is not None else "n/a"
        print(
            "  Human Error (mean/median/P95/failed mean): "
            f"{format_error(errors['human_pose_m']['mean'])}/"
            f"{format_error(errors['human_pose_m']['median'])}/"
            f"{format_error(errors['human_pose_m']['p95'])}/"
            f"{format_error(errors['human_pose_m']['failed_mean'])}"
        )
        print(
            "  Object Error (mean/median/P95/failed mean): "
            f"{format_error(errors['object_surface_m']['mean'])}/"
            f"{format_error(errors['object_surface_m']['median'])}/"
            f"{format_error(errors['object_surface_m']['p95'])}/"
            f"{format_error(errors['object_surface_m']['failed_mean'])}"
        )
        print("  Per-sequence results:")
        for sequence in sequence_summaries:
            semantic_rate = sequence['semantic_rate']
            rate_text = f"{semantic_rate:.2%}" if semantic_rate is not None else "n/a"
            print(
                f"    [EVAL_SEQ] {sequence['sequence']} "
                f"episodes={sequence['episodes']} "
                f"completed={sequence['completed']} semantic={rate_text} "
                f"wrong_contact_steps={sequence['wrong_contact_steps']}"
            )
        print("=" * 72 + "\n")

        if actual_episodes != expected_episodes:
            raise RuntimeError(
                "Strict evaluation cohort incomplete: "
                f"recorded {actual_episodes}, expected {expected_episodes}. "
                "Increase player max_steps or inspect early termination."
            )
    
@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat
