# Using Docker

We've also provided a docker compose configuration to automatically setup the dependencies and requirements necessary to run the code in this repo.

```{note}
This is not a full explanation of Docker or Docker Compose. If you are unfamiliar with Docker, we recommend you read the [official documentation](https://docs.docker.com/get-started/).
```

## Prerequisites

You need to have Docker installed on your machine. You can download it [here](https://docs.docker.com/engine/install/). Additionally, your system must have [cuda](https://developer.nvidia.com/cuda-downloads) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Installation

To build the docker image, run:

```bash
docker compose up -d
```

The first time this is run, this will build two services: `aci` and `vnc`. The `vnc` service allows you to visualize GUIs from within the `aci` container using a docker network bridge. It will then initialize the services to run in the background.

```{tip}
If you change the `Dockerfile` or `docker-compose.yml`, you will need to rebuild the image with `docker compose build`.
```

You can then attach to the `aci` container with:

```bash
docker compose exec -it aci bash
```

The working directory within the `aci` container is `/home/aci/aci`. You then need to install the `aci` package in the container. You only need to do this once after building.

```bash
pip install -e .
```

## Running

You can then run the code as you would on your local machine. For example, to run the training script:

```bash
python cambrian/main.py --eval example=detection_optimal env.renderer.render_modes='[human]' env.renderer.save_mode=NONE trainer.max_episode_steps=10000 eval_env.n_eval_episodes=1
```

The display window should then be visible from within the `vnc` container. You can connect to the VNC server in your browser. It's likely the server will be located at `http://localhost:8080`.

```{note}
It's possible that port `8080` was in use, in which case the VNC server may be located on any port between `8080` and `8099`. You can find the port by running `docker compose ps`.
```
