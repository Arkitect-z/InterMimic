from enum import Enum
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
        # Evaluation only works with stateInit "Start"
        state_init_is_start = (state_init == "Start")
        self.enable_evaluation = cfg['env'].get('enableEvaluation', False) and state_init_is_start
        if cfg['env'].get('enableEvaluation', False) and not state_init_is_start:
            print(f"Warning: Evaluation is disabled because stateInit is '{state_init}' (must be 'Start')")
        motion_file = os.listdir(self.motion_file)
        self.motion_file = sorted([os.path.join(self.motion_file, data_path) for data_path in motion_file if data_path.split('_')[0] in cfg['env']['dataSub']])

        # Parse dual-object names from filename: sub1_ObjA+ObjB_seqname.pt
        self._motion_obj_pairs = []  # [(obj1, obj2)] per motion
        unique_obj_set = set()
        for path in self.motion_file:
            combined = os.path.basename(path).split('_')[-2]  # e.g. "CupBlue+KettleGreen"
            if '+' in combined:
                o1, o2 = combined.split('+', 1)
            else:
                o1, o2 = combined, combined
            self._motion_obj_pairs.append((o1, o2))
            unique_obj_set.update([o1, o2])

        # Construct device string before super().__init__()
        if device_type == "cuda" or device_type == "GPU":
            self._init_device = "cuda:" + str(device_id)
        else:
            self._init_device = "cpu"

        self.object_name = sorted(list(unique_obj_set))  # unique object types
        self.obj1_id = to_torch([self.object_name.index(p[0]) for p in self._motion_obj_pairs], dtype=torch.long).to(self._init_device)
        self.obj2_id = to_torch([self.object_name.index(p[1]) for p in self._motion_obj_pairs], dtype=torch.long).to(self._init_device)
        # Backward compat: object_id points to obj1 for legacy code paths
        self.object_id = self.obj1_id
        # With dual objects, every motion contains both objects — all motions are valid
        self.obj2motion = torch.ones((len(self.object_name), len(self._motion_obj_pairs)), dtype=torch.bool).to(self._init_device)
        self.robot_type = cfg['env']['robotType']
        self.object_density = cfg['env']['objectDensity']
        # 2-object hoi_data: human(7+306+676) + obj1(13) + obj2(13) + ig1(156) + ig2(156) + contact(52+1+1)
        self.ref_hoi_obs_size = 7 + 51 * 6 + 52 * 13 + 13 * 2 + 52 * 3 * 2 + 52 + 2
        self.num_motions = len(self.motion_file)
        self.dataset_index = to_torch([int(data_path.split('/')[-1].split('_')[0][3:]) for data_path in self.motion_file], dtype=torch.long).to(self._init_device)

        self._preload_table_info()

        if self.play_dataset:
            sim_params.gravity = gymapi.Vec3(0, 0, 0)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

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
        self._hand_fail_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._diag_file = None
        if self.num_envs <= 16:
            diag_path = os.path.join(os.getcwd(), 'diag_output.txt')
            self._diag_file = open(diag_path, 'w')
            print(f"[DIAG] Writing per-step diagnostics to {diag_path}")
        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Use max_episode_length for buffer size (rollout_length may be auto-adjusted later)
        buf_len = max(cfg['env']['rolloutLength'], cfg['env']['episodeLength'])
        self._curr_reward = torch.zeros([self.num_envs, buf_len], device=self.device, dtype=torch.float)
        self._sum_reward = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._curr_state = torch.zeros([self.num_envs, buf_len, 345], device=self.device, dtype=torch.float)
        self._build_target_tensors()

        return

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
        """Compute table surface height from frame-0 object pose for both objects."""
        self._table_info = {}  # key: obj_name -> table info
        from scipy.spatial.transform import Rotation as R

        if not self.motion_file:
            return
        data = torch.load(self.motion_file[0], weights_only=False)

        obj_specs = [
            (self._motion_obj_pairs[0][0], 318, 321, 325),  # obj1: pos 318:321, rot 321:325
            (self._motion_obj_pairs[0][1], 325, 328, 332),  # obj2: pos 325:328, rot 328:332
        ]
        for obj_name, pos_s, pos_e, rot_e in obj_specs:
            obj_pos_0 = data[0, pos_s:pos_e].numpy()
            obj_rot_0 = data[0, pos_e:rot_e].numpy()

            obj_file = resolve_data_path("assets", "objects", "objects", obj_name, obj_name + ".obj")
            if not os.path.exists(str(obj_file)):
                continue
            mesh = trimesh.load(str(obj_file), force='mesh')
            rot = R.from_quat(obj_rot_0)
            verts_world = rot.apply(mesh.vertices) + obj_pos_0
            table_top_z = verts_world[:, 2].min()

            half_x = max(mesh.extents[0] * 0.5, 0.03)
            half_y = max(mesh.extents[1] * 0.5, 0.03)
            self._table_info[obj_name] = {
                'table_top_z': float(table_top_z),
                'half_x': half_x,
                'half_y': half_y,
                'init_x': float(obj_pos_0[0]),
                'init_y': float(obj_pos_0[1]),
                'reset_dist': float(mesh.extents.max() * 0.3),
            }

    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):

        hoi_datas = []
        hoi_refs = []
        if type(motion_file) != type([]):
            motion_file = [motion_file]
        max_episode_length = []
        # Process data on CPU first, then move to GPU at the end
        object_points_cpu = self.object_points.cpu()
        object_id_cpu = self.object_id.cpu()

        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = torch.load(data_path)[startk:]
            loaded_dict['hoi_data'] = hoi_data.detach()  # Keep on CPU for processing


            max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = 30.

            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot'])
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])),loaded_dict['root_rot_vel']),dim=0)

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
            o1_exp = torch_utils.quat_to_exp_map(loaded_dict['obj1_rot'])
            loaded_dict['obj1_rot_vel'] = torch.cat((torch.zeros((1, 3)), (o1_exp[1:] - o1_exp[:-1]) * self.fps_data), dim=0)

            # --- Object 2 ---
            loaded_dict['obj2_pos'] = loaded_dict['hoi_data'][:, 325:328].clone()
            loaded_dict['obj2_pos_vel'] = torch.cat((torch.zeros((1, 3)), (loaded_dict['obj2_pos'][1:] - loaded_dict['obj2_pos'][:-1]) * self.fps_data), dim=0)
            loaded_dict['obj2_rot'] = loaded_dict['hoi_data'][:, 328:332].clone()
            o2_exp = torch_utils.quat_to_exp_map(loaded_dict['obj2_rot'])
            loaded_dict['obj2_rot_vel'] = torch.cat((torch.zeros((1, 3)), (o2_exp[1:] - o2_exp[:-1]) * self.fps_data), dim=0)

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
            loaded_dict['body_rot'] = loaded_dict['hoi_data'][:, 386:386+52*4].clone()

            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['body_rot'].view(-1, 4)).view(-1, 52*3)
            loaded_dict['body_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['body_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_rot_vel'].shape[-1])),loaded_dict['body_rot_vel']),dim=0)

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
            self._max_execution_steps = torch.zeros([self.num_motions], device=self.hoi_refs.device, dtype=torch.long)
            self._human_pose_error_per_seq_step = torch.ones([self.num_motions, max_length], device=self.hoi_refs.device, dtype=torch.float) * 1e6
            self._object_pose_error_per_seq_step = torch.ones([self.num_motions, max_length], device=self.hoi_refs.device, dtype=torch.float) * 1e6
            self._best_human_pose_error_per_seq = torch.ones([self.num_motions], device=self.hoi_refs.device, dtype=torch.float) * 1e6
            self._best_object_pose_error_per_seq = torch.ones([self.num_motions], device=self.hoi_refs.device, dtype=torch.float) * 1e6
            # Track visit counts for balanced sampling
            self._sequence_visit_count = torch.zeros([self.num_motions], device=self.hoi_refs.device, dtype=torch.long)

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
        self._load_target_asset()
        self._has_table = bool(self._table_info)

        # Override parent's aggregate sizing for dual-object envs:
        # 2 objects + 2 tables = 4 extra bodies; shapes include convex hulls for both objects
        self._extra_agg_bodies = 4
        self._extra_agg_shapes = 200

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
            density = self.object_density
        
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

            mesh_obj = trimesh.load(str(obj_file), force='mesh')
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            object_points = to_torch(object_points - center)
            

            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))

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
        return handle

    def _create_table_actor(self, env_ptr, env_id, obj_name, col_group, segmentation_id):
        """Create table actor for the given object (after all object actors are created)."""
        info = self._table_info.get(obj_name)
        if info is None or obj_name not in self._table_assets:
            return
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(info['init_x'], info['init_y'],
                                   info['table_top_z'] - self._table_thickness / 2)
        table_h = self.gym.create_actor(
            env_ptr, self._table_assets[obj_name], table_pose,
            f"table_{obj_name}", col_group, 1, segmentation_id,
        )
        self.gym.set_rigid_body_color(env_ptr, table_h, 0, gymapi.MESH_VISUAL,
                                      gymapi.Vec3(0.4, 0.3, 0.2))

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 1 if self.play_dataset else 0
        seg_id = 0

        pair = self._motion_obj_pairs[env_id % len(self._motion_obj_pairs)]
        # Create all object actors FIRST so they occupy consecutive actor indices
        h1 = self._create_object_actor(env_ptr, env_id, pair[0], col_group, col_filter, seg_id)
        h2 = self._create_object_actor(env_ptr, env_id, pair[1], col_group, col_filter, seg_id)
        self._target_handles.append((h1, h2))
        # Create tables after both objects — tables are fixed and don't need state tensors
        self._create_table_actor(env_ptr, env_id, pair[0], col_group, seg_id)
        self._create_table_actor(env_ptr, env_id, pair[1], col_group, seg_id)

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
        return
    
    def _reset_target(self, env_ids):
        d, r, t = self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids]
        for st, prefix in [(self._target_states_1, 'obj1'), (self._target_states_2, 'obj2')]:
            st[env_ids, :3] = self.extract_ref_component(f'{prefix}_pos', d, r, t)
            st[env_ids, 3:7] = self.extract_ref_component(f'{prefix}_rot', d, r, t)
            st[env_ids, 7:10] = self.extract_ref_component(f'{prefix}_pos_vel', d, r, t)
            st[env_ids, 10:13] = self.extract_ref_component(f'{prefix}_rot_vel', d, r, t)

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        ids = torch.cat([self._tar_actor_ids_1[env_ids], self._tar_actor_ids_2[env_ids]])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(ids), len(ids),
        )

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

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
                # Get valid motion indices for this object type
                obj_type = env_idx % len(self.object_name)
                valid_motions = torch.where(self.obj2motion[obj_type] == 1)[0]

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
            # Original random sampling for training
            i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)

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
        i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)
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
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
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
        self._curr_obs[:] = self.build_hoi_observations(
            self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
            self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:, 0, :],
            self._dof_pos, self._dof_vel, self._rigid_body_pos,
            self._local_root_obs, self._root_height_obs, self._dof_obs_size,
            self._target_states_1, self._target_states_2,
            self._tar_contact_forces_1, self._tar_contact_forces_2,
            self._contact_forces,
            self.object_points[self.obj1_id[self.data_id]],
            self.object_points[self.obj2_id[self.data_id]],
            self._rigid_body_rot, self._rigid_body_vel, self._rigid_body_ang_vel,
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
            reset_id = torch.where(self.reset_buf)[0]
            flag = False
            for id in reset_id:
                seq_id = self.data_id[id]
                curr_steps = self.progress_buf[id]

                # Since stateInit is "Start", we average from 1 to curr_steps (skip index 0 which is never computed)
                if self._max_execution_steps[seq_id] < curr_steps:
                    self._max_execution_steps[seq_id] = curr_steps
                    # Average from index 1 onwards (index 0 is the initial state, no reward computed)
                    self._best_human_pose_error_per_seq[seq_id] = self._human_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    self._best_object_pose_error_per_seq[seq_id] = self._object_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    flag = True
                elif self._max_execution_steps[seq_id] == curr_steps:
                    curr_human_error = self._human_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    curr_object_error = self._object_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    if self._best_human_pose_error_per_seq[seq_id] + self._best_object_pose_error_per_seq[seq_id] > curr_human_error + curr_object_error:
                        self._best_human_pose_error_per_seq[seq_id] = curr_human_error
                        self._best_object_pose_error_per_seq[seq_id] = curr_object_error
                        flag = True

            if (self._max_execution_steps >= 1).all() and flag:
                avg_execution_steps = self._max_execution_steps[self._max_execution_steps > 0].float().mean()
                avg_human_error = self._best_human_pose_error_per_seq[self._best_human_pose_error_per_seq < 1e5].mean()
                avg_object_error = self._best_object_pose_error_per_seq[self._best_object_pose_error_per_seq < 1e5].mean()
                success_count = torch.sum(self._max_execution_steps - (self.max_episode_length - 1) >= 0)
                success_rate = success_count.float() / self.max_episode_length.shape[0]

                print('=' * 60)
                print('EVALUATION METRICS:')
                print(f'  Average Execution Steps: {avg_execution_steps:.2f}')
                print(f'  Average Human Pose Error: {avg_human_error:.4f}')
                print(f'  Average Object Pose Error: {avg_object_error:.4f}')
                print(f'  Success Rate: {success_rate:.2%} ({success_count}/{self.max_episode_length.shape[0]})')
                print('=' * 60)

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

    # L_Wrist DOFs: joint 16 (0-indexed from L_Hip) -> DOF indices 48,49,50
    # R_Wrist DOFs: joint 34 -> DOF indices 102,103,104
    _WRIST_DOF_IDX = [48, 49, 50, 102, 103, 104]

    def _init_residual_scale(self):
        self._residual_scale_per_dof = torch.full((153,), 0.3, device=self.device)
        self._residual_scale_per_dof[self._FINGER_DOF_IDX] = 0.6
        self._residual_scale_per_dof[self._WRIST_DOF_IDX] = 0.5

    def _action_to_pd_targets(self, action):
        """Residual control: body=0.3, finger=0.6 for extra grasping freedom."""
        if not hasattr(self, '_residual_scale_per_dof'):
            self._init_residual_scale()
        ref_dof = self.extract_data_component('dof_pos', obs=self._curr_ref_obs)
        return ref_dof + self._residual_scale_per_dof * self._pd_action_scale * action

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

        # Per-object ro_safe: only clamp the object whose hand is in contact
        ro1, ro2 = self._ro1, self._ro2
        ro1_safe = torch.where(left_any > 0.5, torch.clamp(ro1, min=0.5), ro1)
        ro2_safe = torch.where(right_any > 0.5, torch.clamp(ro2, min=0.5), ro2)
        ro_safe = ro1_safe * ro2_safe

        # --- Wrist error + reset (tighter: 15cm / 20 frames) ---
        sim_body_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(-1, 52, 3)
        ref_body_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(-1, 52, 3)
        left_wrist_err = (sim_body_pos[:, 17] - ref_body_pos[:, 17]).norm(dim=-1)
        right_wrist_err = (sim_body_pos[:, 36] - ref_body_pos[:, 36]).norm(dim=-1)
        hand_fail = (torch.max(left_wrist_err, right_wrist_err) > 0.15).float()
        self._hand_fail_counter = (self._hand_fail_counter + hand_fail) * hand_fail
        self._hand_fail_reset = (self._hand_fail_counter > 20) & (self.progress_buf > self.start_times + 10)

        # --- Object trajectory reset (contact phase: obj off by >30% of its size for 20 frames) ---
        obj1_pos = self.extract_data_component('obj1_pos', obs=self._curr_obs)
        ref_obj1_pos = self.extract_data_component('obj1_pos', obs=self._curr_ref_obs)
        obj2_pos = self.extract_data_component('obj2_pos', obs=self._curr_obs)
        ref_obj2_pos = self.extract_data_component('obj2_pos', obs=self._curr_ref_obs)
        obj1_name = self._motion_obj_pairs[0][0]
        obj2_name = self._motion_obj_pairs[0][1]
        obj1_thresh = self._table_info.get(obj1_name, {}).get('reset_dist', 0.05)
        obj2_thresh = self._table_info.get(obj2_name, {}).get('reset_dist', 0.05)
        obj_fail = torch.max(
            left_any * ((obj1_pos - ref_obj1_pos).norm(dim=-1) > obj1_thresh).float(),
            right_any * ((obj2_pos - ref_obj2_pos).norm(dim=-1) > obj2_thresh).float(),
        )
        self._obj_fail_counter = (self._obj_fail_counter + obj_fail) * obj_fail
        obj_fail_reset = (self._obj_fail_counter > 20) & (self.progress_buf > self.start_times + 10)
        self._hand_fail_reset = self._hand_fail_reset | obj_fail_reset

        # --- Additive wrist tracking bonus (broad gradient, strong at 5-15cm) ---
        wrist_bonus = left_any * torch.exp(-5.0 * left_wrist_err) \
                    + right_any * torch.exp(-5.0 * right_wrist_err)

        # --- Grasp success bonus (physical contact during reference contact phase) ---
        sim_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        grasp_bonus = left_any * (sim_contact[:, 17:33] > 0.1).any(dim=-1).float() \
                    + right_any * (sim_contact[:, 36:52] > 0.1).any(dim=-1).float()

        # --- Total reward ---
        self.rew_buf[:] = rb * ro_safe * rig * rcg \
                        + 0.05 * self._r_finger \
                        + 0.3 * wrist_bonus \
                        + 0.05 * grasp_bonus

        kinematic_reset = torch.logical_or(human_reset, object_reset)
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        index = torch.arange(self._curr_reward.shape[0])
        self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
        self._sum_reward[index] += self.rew_buf
        self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
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

        self.extras['sub_rewards'] = {
            'rb': rb.mean().item(),
            'ro': ro.mean().item(),
            'ro_safe': ro_safe.mean().item(),
            'ro1_safe': ro1_safe.mean().item(),
            'ro2_safe': ro2_safe.mean().item(),
            'rig': rig.mean().item(),
            'rcg': rcg.mean().item(),
            'r_finger': self._r_finger.mean().item(),
            'wrist_bonus': wrist_bonus.mean().item(),
            'grasp_bonus': grasp_bonus.mean().item(),
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
                    f"rb={rb[ei]:.3f} ro={ro[ei]:.3f} ro_s={ro_safe[ei]:.3f} "
                    f"rig={rig[ei]:.3f} rcg={rcg[ei]:.3f} "
                    f"r_fin={self._r_finger[ei]:.3f} rew={self.rew_buf[ei]:.3f} "
                    f"wrist_err={left_wrist_err[ei]:.4f} hand_obj={left_wrist_obj:.4f} "
                    f"hfail={self._hand_fail_counter[ei]:.0f} "
                    f"fin_err={finger_err[ei]:.4f}\n"
                )
                if self.reset_buf[ei] > 0:
                    self._diag_file.write(f"  >>> RESET env{ei} after {t - st} steps\n")
            self._diag_file.flush()

        if self.enable_evaluation:
            self._human_pose_error_per_seq_step[self.data_id, self.progress_buf] = human_error
            self._object_pose_error_per_seq_step[self.data_id, self.progress_buf] = object_error

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
    
    def compute_cg_reward(self, w):    
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        left_contact_hand_ids = list(range(17, 33))
        
        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, left_contact_hand_ids].clone()
        left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        # any()-based contact: reward if ANY hand body touches the object
        ecg_left = ref_left_contact_hand_any * (1.0 - left_hand_contact_any.squeeze(-1))
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left*w['cg_hand'])) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)

        right_contact_hand_ids = list(range(36, 52))
        
        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, right_contact_hand_ids].clone()
        right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        contact_reset = torch.cat([ 
                                torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(-1), 
                                torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1),
                                ], dim=-1)
        
        ecg_right = ref_right_contact_hand_any * (1.0 - right_hand_contact_any.squeeze(-1))
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right*w['cg_hand'])) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)
        
        rcg_hand = rcg_left * rcg_right

        other_ids = [i for i in range(len(self.contact_bodies)) if i not in left_contact_hand_ids and i not in right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other*w['cg_other'])
        
        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all*w['cg_all'])

        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

        rcg = rcg_hand*rcg_other*rcg_all*contact_energy
        return rcg, contact_reset
    
    def play_dataset_step(self, time):

        t = time
        if t == 0:
            self.data_id = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in range(self.num_envs)], device=self.device, dtype=torch.long)
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

    def print_final_eval_summary(self):
        """Print final evaluation summary at the end of inference"""
        if not self.enable_evaluation:
            return

        evaluated_mask = self._max_execution_steps >= 1
        num_evaluated = evaluated_mask.sum()

        if num_evaluated == 0:
            print("=" * 60)
            print("WARNING: No sequences were evaluated!")
            print("Consider increasing --max_steps in the evaluation script")
            print("=" * 60)
            return

        avg_execution_steps = self._max_execution_steps[evaluated_mask].float().mean()
        avg_human_error = self._best_human_pose_error_per_seq[evaluated_mask].mean()
        avg_object_error = self._best_object_pose_error_per_seq[evaluated_mask].mean()
        success_count = torch.sum(self._max_execution_steps[evaluated_mask] - (self.max_episode_length[evaluated_mask] - 1) >= 0)
        success_rate = success_count.float() / num_evaluated

        # Visit statistics
        min_visits = self._sequence_visit_count.min().item()
        max_visits = self._sequence_visit_count.max().item()
        avg_visits = self._sequence_visit_count.float().mean().item()

        print("\n" + "=" * 60)
        print("FINAL EVALUATION SUMMARY:")
        print(f"  Sequences Evaluated: {num_evaluated}/{self.num_motions} ({100.0 * num_evaluated / self.num_motions:.1f}%)")
        print(f"  Sequence Visits - Min: {min_visits}, Max: {max_visits}, Avg: {avg_visits:.1f}")
        print(f"  Average Execution Steps: {avg_execution_steps:.2f}")
        print(f"  Average Human Pose Error: {avg_human_error:.4f}")
        print(f"  Average Object Pose Error: {avg_object_error:.4f}")
        print(f"  Success Rate: {success_rate:.2%} ({success_count}/{num_evaluated})")
        print("=" * 60 + "\n")
    
@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat
