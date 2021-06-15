import numpy as np
import roboverse.bullet as bullet
from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS

class DrawerOpenGrasp:

    def __init__(self, env, pick_height_thresh=-0.23, pick_point_noise=0.00, pick_point_z=-0.31):
        self.env = env
        self.xyz_action_scale = 7.0
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.04
        self.ending_height_thresh = 0.2
        
        self.pick_height_thresh = pick_height_thresh
        self.pick_point_noise = pick_point_noise
        self.pick_point_z = pick_point_z
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.drawer_never_opened = True
        offset_coeff = (-1) ** (1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.01])

        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]
        print(self.env.num_objects)
        print(self.object_to_target)
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        self.pick_point += np.random.normal(scale=self.pick_point_noise, size=(3,))
        self.pick_point[2] = self.pick_point_z + np.random.normal(scale=0.01)
        self.t_upward = 4
        self.reached_grasp = False

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        done = False

        if (gripper_handle_xy_dist > self.gripper_xy_dist_thresh
                and not self.env.is_drawer_open()) and not self.reached_grasp:
            print('xy - approaching handle')
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (gripper_handle_dist > self.gripper_dist_thresh
                and not self.env.is_drawer_open()) and not self.reached_grasp:
            # moving down toward handle
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not self.env.is_drawer_open() and not self.reached_grasp:
            print("opening drawer")
            x_command = (-1) ** (1 - self.env.left_opening)
            action_xyz = np.array([x_command, 0, 0])
            # action = np.asarray([0., 0., 0.7])
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (np.abs(ee_pos[2] - self.ending_height_thresh) >
                self.gripper_dist_thresh) and self.t_upward >= 0:
            # print("Lift upward")
            self.drawer_never_opened = False
            action_xyz = np.array([0, 0, 0.7])  # force upward action
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            self.t_upward -= 1
        else:
            self.reached_grasp=True
            if gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
                # move near the object
                print('move to object')
                print(self.pick_point)
                action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
                xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
                if xy_diff > 0.03:
                    action_xyz[2] = 0.0
                action_angles = [0., 0., 0.]
                action_gripper = [0.]
            elif self.env.is_gripper_open:
                # near the object enough, performs grasping action
                print('grasp')
                action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
                action_angles = [0., 0., 0.]
                action_gripper = [-0.7]
            elif not object_lifted:
                # lifting objects above the height threshold for picking
                action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
                action_angles = [0., 0., 0.]
                action_gripper = [0.]
            else:
                # Hold
                action_xyz = (0., 0., 0.)
                action_angles = [0., 0., 0.]
                action_gripper = [0.]

        agent_info = dict(done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info