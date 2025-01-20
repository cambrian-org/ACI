# Pitfalls

This page outlines common issues that may arise when using the ACI framework.

## `RuntimeError: invalid value for environment variable MUJOCO_GL: egl`

This error may occur when trying to run an environment headless using the `egl` backend. To fix, you can either explicitly set `MUJOCO_GL` to a supported value on your system, as found [here](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/gl_context.py), or you can install the EGL backend for OpenGL. Installation depends on your system.
