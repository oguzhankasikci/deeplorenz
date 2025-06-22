import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from lorenzsystem  import LorenzClass


# 1.  Simulation parameters
# ##########################################
SIGMA, BETA, RHO = 10.0, 8.0 / 3.0, 28.0     # lorenz system parameters
DT,  N_STEPS   = 0.009, 10_000               # For particles time difference and  number of steps 
ATTR_DT, ATTR_STEPS = 0.01, 20_000           # For Attractor time difference and number of steps
ATTR_INITIAL = [-8., 8., 20.]                # Attractor initial point
DRAW_EVERY     = 5                           # animation stride
N_PART         = 500                         # particles per colour
XLIM, YLIM, ZLIM = (-20, 20), (-30, 30), (0, 50)
SAVE = 'no'


# 2.  Initialise particle clouds
# ##########################################
rng = np.random.default_rng(0)
red_cloud  = np.array([[-4.,  8., 25.]]) + 0.5 * rng.standard_normal((N_PART, 3))
blue_cloud = np.array([[ 4., -8., 25.]]) + 0.5 * rng.standard_normal((N_PART, 3))
particles  = np.vstack((red_cloud, blue_cloud))
colors     = np.array(['tab:red'] * N_PART + ['tab:orange'] * N_PART)

#  frames for particle animation
n_frames   = N_STEPS // DRAW_EVERY + 1
frames_xyz = np.empty((n_frames, particles.shape[0], 3))
frames_xyz[0] = particles

# 3. Background  Lorenz Attractor
# ##########################################
lc = LorenzClass()
lc.set_initial_value(ATTR_INITIAL)
lc.set_parameters([SIGMA, BETA, RHO])
lc.set_time_grid(t0=0,tf= ATTR_DT * ATTR_STEPS, n_steps= ATTR_STEPS)
attractor = lc.solve()


# 4.  Integrate particle motion
# ##########################################
for s in range(1, N_STEPS+1):
    particles += DT * lc.lorenz(particles.T).T
    if s % DRAW_EVERY == 0:
        frames_xyz[s // DRAW_EVERY] = particles


# 5. Set up dark‑background plot & widgets
# ##########################################
plt.style.use("dark_background")   # global dark theme

fig = plt.figure(figsize=(9, 7), facecolor='black')
ax  = fig.add_subplot(111, projection='3d', proj_type='ortho')
ax.set_facecolor('black')          # 3‑D pane

# make panes & grids dark
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0, 0, 0, 1))
    axis._axinfo["grid"]['color'] = (0.0, 0.0, 0.0, 1)

ax.set(xlim=XLIM, ylim=YLIM, zlim=ZLIM,
       xlabel='x', ylabel='y', zlabel='z')

# background wireframe attractor
ax.plot(*attractor.T, lw=0.3, color='LightSkyBlue', alpha=0.6, zorder=1)

# particle scatter
scat = ax.scatter(*frames_xyz[0].T, c=colors, s=6, zorder=2)

# 6. Animation control logic
# ##########################################
state = {"frame": 0, "running": True}

def update(frame_i):
    idx   = state["frame"]
    time  = idx * DRAW_EVERY * DT          # seconds
    scat._offsets3d = frames_xyz[idx].T
    ax.set_title(f"Time = {time:5.2f} s",  # ← add “ s” here
                 fontsize=12, color='w')
    if state["running"]:
        state["frame"] = (idx + 1) % n_frames
    return scat,


ani = animation.FuncAnimation(fig, update, interval=30, blit=False)


# 7.  Buttons  (dark-grey faces + lighter hover)
# ##########################################
button_ax = {}
button_ax['play'] = plt.axes([0.15, 0.01, 0.08, 0.05])
button_ax['back'] = plt.axes([0.26, 0.01, 0.08, 0.05])
button_ax['fwd']  = plt.axes([0.37, 0.01, 0.08, 0.05])
button_ax['quit'] = plt.axes([0.79, 0.01, 0.10, 0.05])
button_ax['reset'] = plt.axes([0.48, 0.01, 0.10, 0.05])


# Darker-grey buttons
BTN_FC   = '#333333'   # normal face colour
BTN_HFC  = '#555555'   # hover face colour

btn_play = Button(button_ax['play'],
                  '❚❚' if state["running"] else '►',
                  color=BTN_FC, hovercolor=BTN_HFC)

btn_back = Button(button_ax['back'], '«',
                  color=BTN_FC, hovercolor=BTN_HFC)

btn_fwd  = Button(button_ax['fwd'],  '»',
                  color=BTN_FC, hovercolor=BTN_HFC)

btn_quit = Button(button_ax['quit'], 'Quit',
                  color=BTN_FC, hovercolor=BTN_HFC)

btn_reset = Button(button_ax['reset'], 'Reset',
                   color=BTN_FC, hovercolor=BTN_HFC)

# Callbacks (unchanged)
def toggle_play(event):
    state["running"] = not state["running"]
    btn_play.label.set_text('❚❚' if state["running"] else '►')

def step_back(event):
    if not state["running"]:
        state["frame"] = (state["frame"] - 1) % n_frames
        update(None)
        fig.canvas.draw_idle()

def step_fwd(event):
    if not state["running"]:
        state["frame"] = (state["frame"] + 1) % n_frames
        update(None)
        fig.canvas.draw_idle()

def reset_anim(event):
    # go to first frame & make sure we’re playing
    state["frame"]  = 0
    state["running"] = False
    btn_play.label.set_text('❚❚')          # show “pause” icon
    update(None)                            # redraw frame 0 immediately
    fig.canvas.draw_idle()


def quit_app(event):
    plt.close(fig)

btn_play.on_clicked(toggle_play)
btn_back.on_clicked(step_back)
btn_fwd .on_clicked(step_fwd)
btn_quit.on_clicked(quit_app)
btn_reset.on_clicked(reset_anim)

# 8.  Save
# ##########################################
#  Example: `python file.py save` comman will save the animation.
#  without second option 'save' the file will just run wihtout saving  

if SAVE == 'yes':
    ani.save('lorenz_mixing_dark.mp4', writer='ffmpeg', dpi=200)
    print('Saved lorenz_mixing_dark.mp4')
elif SAVE == 'no':
    print('Run without saving the animation.')
else:
    print('Invalid option. Use "yes" or "no".')

plt.tight_layout()
plt.show()
