# Visualizing the Env

There is a runner in {src}`env.py <cambrian/envs/env.py>` that will visualize the world. You have a few run options:

## Display Methods

### Interactive + Display

Run with a custom visualization viewer in birds-eye view mode. This is interactive,
so you can pan the renderer around using the mouse.

```bash
python cambrian/envs/env.py run_renderer exp=<EXPERIMENT> env.renderer.render_modes="[human]"
```

```{eval-rst}
.. note::

    You may need to set ``MUJOCO_GL=glfw`` explicitly to have this work properly.

    .. code-block:: bash

        MUJOCO_GL=glfw python cambrian/envs/env.py run_renderer exp=<EXPERIMENT> env.renderer.render_modes="[human]"

```

### Non-interactive + Headless

Run the custom viewer but headless and save the output:

```bash
python cambrian/envs/env.py run_renderer exp=<EXPERIMENT> --record <OPTIONAL_TOTAL_TIMESTEPS>
```

### Interactive + Display

Run with the built-in mujoco viewer. You will need to scroll out to see the full view:

```bash
python cambrian/envs/env.py run_mj_viewer exp=<EXPERIMENT>
```

## Viewer Shortcuts

### Custom Viewer

This is a custom viewer that we use for debugging. There are a few shortcuts you can
use:

- `R`: Reset the environment
- `S`: Screenshot (saved as `screenshot.png`)
- `Tab`: Switch cameras in the main view
- `Space`: Pause the simulation
- `Exit`: Close the window

### Mujoco Viewer

Hover over an option on the left side and right click to show all the shortcuts. `Q` will visualize the cameras and their frustums. You can also just click it under **Rendering**.
