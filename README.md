<h1 align="center">
  Learning Versatile Humanoid Manipulation with Touch Dreaming
</h1>
<p align="center">
  <a href="https://yaruniu.com/" target="_blank">Yaru Niu</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
  Zhenlong Fang<sup>1</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://wudicbh.github.io/" target="_blank">Binghong Chen</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://www.shuai-zhou.com/" target="_blank">Shuai Zhou</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://revanthsenthil.github.io/" target="_blank">Revanth Senthilkumaran</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
  <br />
  <a href="https://haozhang-thu.github.io/" target="_blank">Hao Zhang</a><sup>1,2</sup>&nbsp;&nbsp;&nbsp;
  Bingqing Chen<sup>3</sup>&nbsp;&nbsp;&nbsp;
  Chen Qiu<sup>3</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://etaic.github.io/" target="_blank">H. Eric Tseng</a><sup>2</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://jonfranc.com/" target="_blank">Jonathan Francis</a><sup>1,3</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://www.meche.engineering.cmu.edu/directory/bios/zhao-ding.html" target="_blank">Ding Zhao</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
  <br />
  <sup>1</sup>Carnegie Mellon University&nbsp;&nbsp;&nbsp;
  <sup>2</sup>UT Arlington&nbsp;&nbsp;&nbsp;
  <sup>3</sup>Bosch Center for AI&nbsp;&nbsp;&nbsp;
  <br />
</p>

<p align="center">
    International Conference on Intelligent Robots and Systems (IROS) 2026<br />
    <a href="https://humanoid-touch-dream.github.io/">Website</a> |
    <a href="https://arxiv.org/pdf/2604.13015">Paper</a>
</p>

<p align="center">
  <img src="imgs/dream_touch_teaser.gif" width="90%" />
  <br />
  <sub>Autonomous policy (HTD) for long-horzion loco-manipulation</sub>
</p>

## Whole-Body Controller

The HTD whole-body controller tracks commanded linear and angular velocities together with torso height and orientation. The Isaac Lab training pipeline, teacher-to-student
distillation, simulation evaluation, and Unitree G1 deployment code are available
in [IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC).
The full training pipeline can be run on a single GPU, and the release includes
example teacher and student policy checkpoints. The [student policy](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC/tree/main/deploy/policy/g1_student) has also been
validated on our physical Unitree G1 robot.

We also include it here as a Git submodule at `htd_wbc/isaaclab_decoupled_wbc`; initialize it after cloning with `git submodule update --init --recursive`.

<p align="center">
  <img src="imgs/htd_wbc_teaser.gif" width="90%" />
  <br />
  <sub>HTD whole-body controller tracking locomotion and torso-pose commands</sub>
</p>

## Release Checklist

- [x] [Lower-body controller training in Isaac Lab and deployment](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC) <img src="https://img.shields.io/badge/status-released-brightgreen" alt="released" height="20" align="absmiddle" />
  - [x] Teacher policy training and evaluation <img src="https://img.shields.io/badge/status-released-brightgreen" alt="released" height="20" align="absmiddle" />
  - [x] Student policy training and evaluation <img src="https://img.shields.io/badge/status-released-brightgreen" alt="released" height="20" align="absmiddle" />
  - [x] Student policy deployment <img src="https://img.shields.io/badge/status-released-brightgreen" alt="released" height="20" align="absmiddle" />
  - [x] Teacher and student policy example checkpoints <img src="https://img.shields.io/badge/status-released-brightgreen" alt="released" height="20" align="absmiddle" />
- [ ] Whole-body teleoperation and data collection <img src="https://img.shields.io/badge/status-on--going-yellow" alt="on-going" height="20" align="absmiddle" />
  - [ ] Teleoperation with Apple Vision Pro <img src="https://img.shields.io/badge/status-on--going-yellow" alt="on-going" height="20" align="absmiddle" />
  - [ ] Simulation support and Pico teleoperation/data collection <img src="https://img.shields.io/badge/status-on--going-yellow" alt="on-going" height="20" align="absmiddle" />
- [ ] HTD policy training and deployment <img src="https://img.shields.io/badge/status-on--going-yellow" alt="on-going" height="20" align="absmiddle" />

<p align="center">
  <img src="imgs/tea_dream.gif" width="90%" />
  <br />
  <sub>Autonomous policy (HTD) with tactile latent dreaming</sub>
  <br /><br />
  <img src="imgs/scoop_dream.gif" width="90%" />
  <br />
  <sub>Autonomous policy (HTD) with force dreaming</sub>
</p>

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{niu2026humanoidtouchdream,
  title={Learning Versatile Humanoid Manipulation with Touch Dreaming},
  author={Niu, Yaru and Fang, Zhenlong and Chen, Binghong and Zhou, Shuai and Senthilkumaran, Revanth Krishna and Zhang, Hao and Chen, Bingqing and Qiu, Chen and Tseng, H. Eric and Francis, Jonathan and Zhao, Ding},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```
