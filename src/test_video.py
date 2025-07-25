import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Grille du champ
X, Y = np.meshgrid(np.linspace(-12, 12, 30), np.linspace(-12, 12, 30))

# Champ stationnaire : rotation autour de l’origine
def vector_field(x, y):
    norm = np.sqrt(x**2 + y**2) + 1e-5
    u = -y / norm
    v = x / norm
    return u, v

U, V = vector_field(X, Y)

# Initialisation de particules aléatoires dans le champ
n_particles = 100
particles_x = np.random.uniform(-12, 12, n_particles)
particles_y = np.random.uniform(-12, 12, n_particles)

fig, ax = plt.subplots()
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_aspect('equal')
ax.set_title("Champ stationnaire avec particules animées")

# Affichage du champ de fond
ax.quiver(X, Y, U, V, color='lightblue', alpha=0.3)

# Particules animées
particles_plot, = ax.plot([], [], 'ko', markersize=2)

# Microswimmer
t = np.linspace(0, 10, 200)
swimmer_x = t * np.cos(t)
swimmer_y = t * np.sin(t)
line, = ax.plot([], [], 'b-', lw=1.5)
point, = ax.plot([], [], 'ro')

def init():
    particles_plot.set_data([], [])
    line.set_data([], [])
    point.set_data([], [])
    return particles_plot, line, point

def update(i):
    global particles_x, particles_y

    # Microswimmer
    line.set_data(swimmer_x[:i], swimmer_y[:i])
    point.set_data([swimmer_x[i]], [swimmer_y[i]])  

    # Mise à jour des particules
    u, v = vector_field(particles_x, particles_y)
    particles_x += 0.2 * u
    particles_y += 0.2 * v

    # Remise dans le cadre
    out = (particles_x < -12) | (particles_x > 12) | (particles_y < -12) | (particles_y > 12)
    particles_x[out] = np.random.uniform(-12, 12, size=out.sum())
    particles_y[out] = np.random.uniform(-12, 12, size=out.sum())

    particles_plot.set_data(particles_x, particles_y)
    return particles_plot, line, point

ani = animation.FuncAnimation(fig, update, frames=len(t),
                              init_func=init, blit=True)

ani.save("champ_avec_particules.mp4", writer='ffmpeg', fps=30, dpi=300)
plt.show()