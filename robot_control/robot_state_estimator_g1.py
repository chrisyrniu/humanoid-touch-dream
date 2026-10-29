import os
from enum import IntEnum
import numpy as np
import pinocchio as pin
import logging_mp
logger_mp = logging_mp.get_logger(__name__)


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5
    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11
    # Waist
    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21
    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


MOTOR_TO_JOINT_NAME = {
    G1_29_JointIndex.kLeftHipPitch: "left_hip_pitch_joint",
    G1_29_JointIndex.kLeftHipRoll: "left_hip_roll_joint",
    G1_29_JointIndex.kLeftHipYaw: "left_hip_yaw_joint",
    G1_29_JointIndex.kLeftKnee: "left_knee_joint",
    G1_29_JointIndex.kLeftAnklePitch: "left_ankle_pitch_joint",
    G1_29_JointIndex.kLeftAnkleRoll: "left_ankle_roll_joint",
    G1_29_JointIndex.kRightHipPitch: "right_hip_pitch_joint",
    G1_29_JointIndex.kRightHipRoll: "right_hip_roll_joint",
    G1_29_JointIndex.kRightHipYaw: "right_hip_yaw_joint",
    G1_29_JointIndex.kRightKnee: "right_knee_joint",
    G1_29_JointIndex.kRightAnklePitch: "right_ankle_pitch_joint",
    G1_29_JointIndex.kRightAnkleRoll: "right_ankle_roll_joint",
    G1_29_JointIndex.kWaistYaw: "waist_yaw_joint",
    G1_29_JointIndex.kWaistRoll: "waist_roll_joint",
    G1_29_JointIndex.kWaistPitch: "waist_pitch_joint",
    G1_29_JointIndex.kLeftShoulderPitch: "left_shoulder_pitch_joint",
    G1_29_JointIndex.kLeftShoulderRoll: "left_shoulder_roll_joint",
    G1_29_JointIndex.kLeftShoulderYaw: "left_shoulder_yaw_joint",
    G1_29_JointIndex.kLeftElbow: "left_elbow_joint",
    G1_29_JointIndex.kLeftWristRoll: "left_wrist_roll_joint",
    G1_29_JointIndex.kLeftWristPitch: "left_wrist_pitch_joint",
    G1_29_JointIndex.kLeftWristyaw: "left_wrist_yaw_joint",
    G1_29_JointIndex.kRightShoulderPitch: "right_shoulder_pitch_joint",
    G1_29_JointIndex.kRightShoulderRoll: "right_shoulder_roll_joint",
    G1_29_JointIndex.kRightShoulderYaw: "right_shoulder_yaw_joint",
    G1_29_JointIndex.kRightElbow: "right_elbow_joint",
    G1_29_JointIndex.kRightWristRoll: "right_wrist_roll_joint",
    G1_29_JointIndex.kRightWristPitch: "right_wrist_pitch_joint",
    G1_29_JointIndex.kRightWristYaw: "right_wrist_yaw_joint",
}


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _orthonormal_basis_from_z_and_x(z_axis: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    z_axis = _normalize(z_axis)
    x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
    if np.linalg.norm(x_proj) < 1e-8:
        x_proj = np.array([1.0, 0.0, 0.0], dtype=float)
        x_proj = x_proj - np.dot(x_proj, z_axis) * z_axis
    x_axis = _normalize(x_proj)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _project_to_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    return v - np.dot(v, plane_normal) * plane_normal


class G1_29_KinematicStateEstimator:
    """Kinematics-based state estimator for G1 whole-body motion.

    Features:
      1) Uses whole-body kinematics with hand joints locked.
      2) Easy relative-pose queries between any supported frames.
      3) Estimates global torso and EE poses in a ground frame.
      4) Assumes both feet are flat on ground.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        urdf_dir: str | None = None,
    ):
        logger_mp.info("Initialize G1_29_KinematicStateEstimator...")

        if urdf_path is None:
            urdf_path = "./assets/g1/g1_body29_hand14_wrist4010.urdf"
        if urdf_dir is None:
            urdf_dir = "./assets/g1/"

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        full_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, urdf_dir)
        # Lock all finger joints to keep the estimator focused on body + arm kinematics.
        hand_joints_to_lock = [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]
        self.robot = full_robot.buildReducedRobot(
            list_of_joints_to_lock=hand_joints_to_lock,
            reference_configuration=pin.neutral(full_robot.model),
        )
        self.model = self.robot.model
        self.data = self.robot.data
        self.q = pin.neutral(self.model)

        # Add end-effector frames for wrists (same convention as IK module)
        self.model.addFrame(
            pin.Frame(
                "L_ee",
                self.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.model.addFrame(
            pin.Frame(
                "R_ee",
                self.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.data = self.model.createData()

        self.frame_ids = {
            "pelvis": self._get_frame_id(["pelvis"]),
            "torso": self._get_frame_id(["torso_link"]),
            "left_foot": self._get_frame_id(["left_ankle_roll_link"]),
            "right_foot": self._get_frame_id(["right_ankle_roll_link"]),
            "left_ee": self.model.getFrameId("L_ee"),
            "right_ee": self.model.getFrameId("R_ee"),
        }


    def _get_frame_id(self, candidate_names: list[str]) -> int:
        for name in candidate_names:
            fid = self.model.getFrameId(name)
            if fid < self.model.nframes:
                return fid
        raise ValueError(f"None of frame names found: {candidate_names}")


    def _build_pin_q_from_motor_q(self, motor_q: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model)
        for motor_idx, joint_name in MOTOR_TO_JOINT_NAME.items():
            jid = self.model.getJointId(joint_name)
            if jid <= 0:
                continue
            jmodel = self.model.joints[jid]
            if jmodel.nq == 1:
                q[jmodel.idx_q] = motor_q[motor_idx]
        return q

    def _forward_kinematics(self, q: np.ndarray):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def _T_frame_in_pelvis(self, frame_name: str) -> np.ndarray:
        return self.data.oMf[self.frame_ids[frame_name]].homogeneous.copy()

    def relative_pose(self, from_frame: str, to_frame: str, q: np.ndarray | None = None) -> np.ndarray:
        """Return T_from_to (pose of `to_frame` in `from_frame`)."""
        if q is not None:
            self._forward_kinematics(q)
        T_from = self._T_frame_in_pelvis(from_frame)
        T_to = self._T_frame_in_pelvis(to_frame)
        return np.linalg.inv(T_from) @ T_to

    def estimate_world_from_feet_flat(self, q: np.ndarray | None = None) -> np.ndarray:
        """Estimate T_world_pelvis under the assumption both feet are flat on ground."""
        if q is not None:
            self._forward_kinematics(q)

        T_P_LF = self._T_frame_in_pelvis("left_foot")
        T_P_RF = self._T_frame_in_pelvis("right_foot")

        R_P_LF, p_P_LF = T_P_LF[:3, :3], T_P_LF[:3, 3]
        R_P_RF, p_P_RF = T_P_RF[:3, :3], T_P_RF[:3, 3]

        # Foot normals/forward in pelvis frame
        z_l = R_P_LF[:, 2]
        z_r = R_P_RF[:, 2]
        x_l = R_P_LF[:, 0]
        x_r = R_P_RF[:, 0]

        # Build R_PW first (world axes expressed in pelvis), then invert to R_WP.
        # Under flat-foot assumption:
        #   world +Z aligns with foot +Z, world +X aligns with foot forward.
        z_world_in_pelvis = _normalize(z_l + z_r)
        x_world_hint_in_pelvis = _normalize(x_l + x_r)
        # R_PW = world axes in pelvis frame
        R_PW = _orthonormal_basis_from_z_and_x(z_world_in_pelvis, x_world_hint_in_pelvis)
        R_WP = R_PW.T

        # Choose world origin at feet midpoint on ground (z=0).
        p_mid = 0.5 * (p_P_LF + p_P_RF)
        # mid point in world frame
        p_mid_w = R_WP @ p_mid
        t_WP = np.array([-p_mid_w[0], -p_mid_w[1], -p_mid_w[2]], dtype=float)
        # the pose of pelvis frame P in world frame W
        T_WP = np.eye(4, dtype=float)
        T_WP[:3, :3] = R_WP
        T_WP[:3, 3] = t_WP
        return T_WP

    def build_pelvis_ground_aligned_frame(self, T_world_pelvis: np.ndarray) -> np.ndarray:
        """Build pelvis-anchored, ground-projected reference frame.

        Origin: pelvis origin projected to ground (same x/y as pelvis, z=0 in world).
        Z-axis: world vertical (ground normal).
        X-axis: pelvis forward axis projected onto horizontal plane.
        Y-axis: leftward axis to complete a right-handed frame.
        """
        z_world = np.array([0.0, 0.0, 1.0], dtype=float)
        R_WP = T_world_pelvis[:3, :3]
        x_pelvis_w = R_WP[:, 0]

        x_proj = _project_to_plane(x_pelvis_w, z_world)
        if np.linalg.norm(x_proj) < 1e-8:
            x_ref = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x_ref = _normalize(x_proj)

        # y = z x x -> leftward when z is up and x is forward.
        y_ref = _normalize(np.cross(z_world, x_ref))

        T_WPr = np.eye(4, dtype=float)
        T_WPr[:3, :3] = np.column_stack([x_ref, y_ref, z_world])
        T_WPr[:3, 3] = np.array(
            [T_world_pelvis[0, 3], T_world_pelvis[1, 3], 0.0],
            dtype=float,
        )
        return T_WPr

    def estimate_state(
        self,
        motor_q: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Estimate kinematic state with relative and global poses.

        Args:
            motor_q: motor q vector (size 29).
            reference_frame: "world" or "pelvis_ground".
        """

        q = self._build_pin_q_from_motor_q(motor_q)
        self._forward_kinematics(q)

        T_WP = self.estimate_world_from_feet_flat()
        T_P_torso = self._T_frame_in_pelvis("torso")
        T_P_LF = self._T_frame_in_pelvis("left_foot")
        T_P_RF = self._T_frame_in_pelvis("right_foot")
        T_P_LEE = self._T_frame_in_pelvis("left_ee")
        T_P_REE = self._T_frame_in_pelvis("right_ee")
        T_WT = T_WP @ T_P_torso
        T_WPr = self.build_pelvis_ground_aligned_frame(T_WP)

        out = {

            # world and pelvis-ground reference transforms
            "T_world_pelvis": T_WP,
            "T_world_pelvis_ground_ref": T_WPr,
            # global poses
            "T_world_torso": T_WT,
            "T_world_left_ee": T_WP @ T_P_LEE,
            "T_world_right_ee": T_WP @ T_P_REE,
            "T_world_left_foot": T_WP @ T_P_LF,
            "T_world_right_foot": T_WP @ T_P_RF,
            # relative poses
            "T_torso_left_ee": np.linalg.inv(T_P_torso) @ T_P_LEE,
            "T_torso_right_ee": np.linalg.inv(T_P_torso) @ T_P_REE,
        }
        return out
