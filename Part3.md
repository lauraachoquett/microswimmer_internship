# Micro-Swimmer Control using Deep Reinforcement Learning - Part 3 - 3D


<p align="center">
    <img src="fig/readme_fig/streamlines.png" width="500"/>
    <br>
    <i>Figure - Streamlines in the retina capillaries</i>
</p>

## 🤖 Reinforcement Learning environment

### Frenet frame 
We introduce the Frenet frame (also known as the TNB frame), which consists of three orthonormal vectors that move along the curve. Given a non-degenerate smooth parametric curve $\mathbf{\gamma}(s) \in \mathbb{R}^3$ parametrized by its arc length $s$, the Frenet frame is defined by:

$$
\mathbf{T}(s) = \frac{d\mathbf{\gamma}}{ds}(s), \qquad
\mathbf{N}(s) = \frac{d\mathbf{T}/ds}{\|d\mathbf{T}/ds\|}, \qquad
\mathbf{B}(s) = \mathbf{T}(s) \times \mathbf{N}(s)
$$ 
Here, $\mathbf{T}(s)$ is the tangent vector, $\mathbf{N}(s)$ is the normal vector, and $\mathbf{B}(s)$ is the binormal vector. These three vectors form a right-handed orthonormal basis.


<p align="center">
    <img src="fig/readme_fig/FrenetTN.png" width="350"/>
    <img src="fig/readme_fig/Frenetframehelix.gif" width="350"/>
    <br>
    <i>Figure - (Left) To do (Right)  Frenet frame along a helix. </i>
</p>

"Although the Frenet frame can easily be computed, its rotation about the tangent of a general spine curve often leads to undesirable twist in motion design or sweep surface modeling." Computation of Rotation Minimizing Frames, Wang et al, 2008

The local curvature can be defined as  : $\kappa  = \|d\mathbf{T}/ds\|$
The Frenet-Serret formulas are : 
* $d\mathbf{T}/ds = \kappa \mathbf{N} $
* $d\mathbf{N}/ds = -\kappa \mathbf{T} + \tau \mathbf{B} $
* $d\mathbf{B}/ds = -\mathbf{N} $

The Frenet frame is not defined at inflection point ($\kappa \approx 0$), to prevent this issue frames are generated with double reflection algorithm from Wang et al. It aims to generate a moving frame such that the rotation between each frame is minimized.

<p align="center">
    <img src="fig/readme_fig/FrenetFrameCurve.png" width="350"/>
    <img src="fig/readme_fig/RMF_Curve.png" width="350"/>
    <br>
    <i>Figure - (Left) Frames generated with Frenet  (Right)  Rotating minimizing frame generated with Wang et al method </i>
</p>



### State

- $\mathbf{X}$ : Position in local frame (w.r.t. the closest path point)
- $\mathbf{V}$ : Velocity in local frame (optional)
- $\mathbf{P}$ : Previous action
- Lookahead : List of the positions and velocities (optional) of the n points following the closest point along the path.

Positions and velocities are expressed in the same local frame, whereas the previous action is provided in the last computed local frame.
### Action 

- Direction $\mathbf{P}$ (2D vector of unit norm)

### Reward 

$$
r_t = -C \cdot \Delta t_{\text{sim}} - \|x_t - x_{\text{target}}\| + \|x_{t-1} - x_{\text{target}}\| - \beta \cdot d
$$

Where:

- $x_{\text{target}}$ : Target position  
- $d$ : Distance to closest point on the path  
- $C \ (m.s^{-1}), \beta$ : Constant weights


## 🏊‍♂️ Training Protocol
### Episode ending conditions
- Agent reaches target within threshold $\delta$
- Episode time exceeds $t_{\text{max}}$

### Path Variants 
The agent is trained to follow a helical path, parametrized by its radius $r$, pitch $p$ and number of turns $n_t$.

$$
\gamma(t) = 
\begin{pmatrix}
R \cos(t) \\
R \sin(t) \\
\frac{p \cdot t}{2\pi}
\end{pmatrix}, \quad t \in [0, 2\pi \cdot n_t]
$$


<p align="center">
    <img src="fig/readme_fig/helix.png" width="300"/>
    <br>
    <i>Figure - helix </i>
</p>

At the beginning of each episode $n$: 
* The radius $r$ is sampled from a uniform distribution :  $r \sim \mathcal{U}([1/8, r_{max}(n)])$
* The pitch $p$ sampled from a uniform distribution : $p \sim \mathcal{U}([p_{min}(n),2.5])$
* The helix is clockwise or counter clockwise every 2 episodes.
Where $r_{max}$ increases lineary from $1/8$ over 2 the course of the training, and $p_{min}$  decreases lineary from 2.5 to $1/8$

### Background Flow Configurations
To enhance robustness, training can include different types of background flows:
  * **Uniform background flow** : $\forall X, \quad \mathbf{u}(\mathbf{X}) = \mathbf{v}$
  * **Rankine vortex** : The velocity components  $(v_r, v_\theta, v_z)$
    f the Rankine vortex, expressed in terms of the cylindrical coordinate system 
    $(r, \theta, z)$, 
    are given by :
    $$
    \begin{aligned}
    v_r &= 0, \\
    v_\theta(r) &= \frac{\Gamma}{2\pi}
    \begin{cases}
    \frac{r}{a^2}, & \text{if } r \leq a, \\
    \frac{1}{r}, & \text{if } r > a,
    \end{cases} \\
    v_z &= 0, \\
    \end{aligned}
    $$


### Random background flow parameters
The parameters of the background flow are randomly sampled at the beginning of each episode as follows:

- **Uniform flow**:
  - Direction vector $\,\mathbf{d} \in \mathbb{R}^3\,$ is sampled uniformly:
    $\mathbf{d} \sim \mathcal{U}([-1,1]^3)$ and normalized
  - Norm of the velocity:
    $\|\mathbf{u}\| \sim \mathcal{U}(0, 0.6)$

- **Rankine vortex**:
  - Center of the vortex $\,\mathbf{c} = (x_c, y_c ,z_c)\,$ is sampled as: $x_c \sim \mathcal{U}(0, 2), \quad y_c \sim \mathcal{U}(0, 1), \quad z_c \sim \mathcal{U}(0, 1)$
  - Core radius: $a \sim \mathcal{U}(0, 1)$
  - Circulation (positive or negative): $\Gamma \sim \mathcal{U}(-1, 1)$

The function `random_bg_parameters()` returns these five values:
```python
def random_bg_parameters(dim):
    dir = np.random.uniform(-1, 1, dim)
    dir = dir / np.linalg.norm(dir)
    norm = np.random.rand() * 0.6

    a = np.random.rand()
    if dim == 3:
        center = [np.random.rand(), 2 * np.random.rand(), np.random.rand()]
    else:
        center = [2 * np.random.rand(), np.random.rand()]
    cir = (np.random.rand() - 0.5) * 2
    return dir, norm, center, a, cir
```

## ⏱️ Agent Evaluation :
After training, agents are evaluated on different set of parameters and a variaty of background flow configurations to assess their generalization ability and robustness. 

### Background Flow Configurations

Three types of background flow are considered during evaluation:

1. **Uniform Background Flow**  
   The agent swims in a constant flow with 6 directions.
   Each direction is tested with a flow norm of $0.5$.

2. **No Background Flow**  
   The environment is evaluated without any background flow (`free` configuration).

3. **Rankine Vortex**  
   The agent is evaluated in a rotational background flow with a Rankine vortex defined by:
   - circulation intensity: $cir = 2$
   - vortex core strength: $a = 0.25$
   - vortex center: $(0, -0.6, 0.2)$
  
The results are saved in JSON format and ranked using: 
```python
rank_agents_by_rewards(results)
```

## 📉 Result on generic path

* Evaluation uniform background velocity with $\mathbf{u} =0.5 \cdot (1, 0, 0)$

<p align="center">
    <img src="fig/readme_fig/eval_with_dir1_05_counter_helix_3D.png" width="500"/>
    <br>
    <i>Figure - Agent evaluation on helix with counter clockwise rotation and uniform background velocity </i>
</p>

* Evaluation rankine vortex 

<p align="center">
    <img src="fig/readme_fig/eval_with_rankine_a_025__cir_3_center_0_06_02_counter_helix.png" width="500"/>
    <br>
    <i>Figure - Agent evaluation on helix with counter clockwise rotation and uniform background velocity </i>
</p>


## 🧭 A* 
The algorithm is similar in 3D 

## 👀 Result for the retina capillaries

<p align="center">
    <img src="fig/readme_fig/Retina_path_3D.png" width="500"/>
    <img src="fig/readme_fig/Retina_path_3D_zoom.png" width="500"/>
    <br>
    <i>Figure - (Left) (Right)  Frenet frame along a helix. </i>
</p>
