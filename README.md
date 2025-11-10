# Micro-Swimmer Control using Deep Reinforcement Learning (Research Internship)
This project aims to train a single Reinforcement Learning (RL) agent capable of following arbitrary trajectories between arbitrary start–goal pairs.
We adopt a two-stage approach:
1.	A flow-aware path-planning module computes feasible trajectories that account for both geometry and background flow.
2.	A RL control policy, trained on local path segments in a reduced-order model, learns to generalize to arbitrary global trajectories.

<p align="center">
    <img src="readme_fig/RL_retina_zoom.pdf" width="500"/>
    <br>
    <i>Figure - Illustrative schematic of a microswimmer in a periodic blood-filled tube. </i>
</p>

The resulting policy is then evaluated in high-fidelity fluid simulations using [**Mirheo**](https://github.com/cselab/Mirheo) and trained with the TD3 algorithm

<p align="center">
    <img src="readme_fig/abf_rbc_4.png" width="500"/>
    <br>
    <i>Figure - Illustrative schematic of a microswimmer in a periodic blood-filled tube. </i>
</p>
