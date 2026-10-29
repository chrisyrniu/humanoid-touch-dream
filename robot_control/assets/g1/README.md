# Unitree G1 robot descriptions

These descriptions and meshes are used by the physical G1 controller's arm IK
and state estimator. The controller selects `g1_body29_hand14_wrist4010.urdf` for
`--g1_type 0` (default) and `g1_body29_hand14_wrist5010.urdf` for `--g1_type 1`.

Both files preserve the original HTD model parameters, adapted from
[Unitree's G1 revision 1.0 with hands](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf).
The filenames identify the wrist motor variants; both retain the original HTD
body parameters.
