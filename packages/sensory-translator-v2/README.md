# REvFLY Python SDK and CLI

Python SDK and CLI for translating joint trajectories into continuous motor rates and replaying them with FlyMimic MuJoCo. The wheel installs the engine and its APIs. The release ZIP supplies the checkpoints, physics assets, and optional local Control Studio.

## Install from PyPI

Once the `revfly` distribution is listed on PyPI, install it for the SDK and CLI.

```bash
python -m pip install revfly
revfly --version
```

For simulations, add the `physics` extra and download the runtime ZIP for the model checkpoints, physics assets, and local Control Studio. The physics workflow was tested on Linux x86-64 with Python 3.11. On Linux, you can install CPU Torch first to avoid downloading CUDA dependencies. Rendering uses the host's OpenGL/EGL libraries. [Runtime ZIP](https://github.com/IlleJiViN/REvFLY/releases/download/v2.0.0/revfly-2.0.0-local-beta.zip)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'revfly[physics]'
revfly install-assets ./revfly-2.0.0-local-beta.zip
revfly verify-assets
```

To switch an existing environment, run `python -m pip uninstall sensory-translator-v2` before installing `revfly`. Both distributions share files and commands.

Assets are stored in `~/.local/share/sensory-translator-v2/bundle` or under `XDG_DATA_HOME`. An explicit `--asset-root` or `SENSORY_TRANSLATOR_V2_ASSET_ROOT` selects a different bundle. Source checkouts with the legacy physics assets keep their checkout defaults. Add `--replace` to install a different verified bundle in place. An extracted `bundle/` directory is also a valid input.

## Install from a GitHub wheel

Download the wheel alongside the runtime ZIP or extract it from the ZIP. Use the same virtual environment as above and replace the PyPI installation command with the wheel command. [GitHub release](https://github.com/IlleJiViN/REvFLY/releases/tag/v2.0.0)

```bash
python -m pip install './revfly-2.0.0-py3-none-any.whl[physics]'
```

## Run a simulation

```bash
revfly run-multijoint --goal-family swing_scaled --scale 0.92 \
  --mode fast --output ./swing-result --render

revfly serve --host 127.0.0.1 --port 8505 --ui \
  --model-id 20261115_v2_fastbridge_lowenergy
```

The first command writes the manifest, physics traces, and top-view video. The second serves the local [Control Studio](http://127.0.0.1:8505/) and the [OpenAPI specification](http://127.0.0.1:8505/openapi.json). For an environment without OpenGL, omit `--render` and set `MUJOCO_GL=disable` for headless inference.

## Use the Python API

```python
from contextlib import closing

from revfly import MultiJointEngine

with closing(MultiJointEngine()) as engine:
    goal = engine.goal_from_family(family="swing_scaled", scale=0.92)
    result = engine.run(goal["target_rad"], mode="fast", output="./swing-result")
    print(result["best_rmse_deg"], result["independent_replay_verified"])
```

Use `EngineV2` for single-axis simulation and `MultiJointEngine` for seven-axis left-front-leg simulation. `EngineV2Config` and `MultiJointEngineConfig` accept an explicit `asset_root` and optional separate `physics_python`. The default multi-joint checkpoint is `20261116_v2_multijoint15_resnorm`. Use `serve --multijoint-checkpoint PATH` or `run-multijoint --checkpoint PATH` to load a different model.

The multi-joint controller commands all 15 muscles. The bundle supports deterministic-interval, refractory Poisson, and direct LIF replay with separately reported simulation RMSE. Premotor connectome replay requires local MaleCNS data plus `pandas` and a Parquet reader such as `pyarrow`.

## Install from a Git tag

Run this in the target virtual environment. Download and install the runtime ZIP separately using `install-assets`.

```bash
python -m pip install 'revfly[physics] @ git+https://github.com/IlleJiViN/REvFLY.git@v2.0.0'
```

## Build and publish a team release

Build the wheel and source distribution from the repository root. Export the runtime files from a checkout that contains the trained checkpoints and FlyMimic assets.

```bash
python -m pip install build
python -m build
revfly export-assets --asset-root . --dest dist/bundle --include-v2-models --include-ui
```

After editing UI sources, build `apps/control_ui` before exporting. The beta ZIP contains the wheel, `bundle/`, this README, third-party notices, the upstream FlyMimic license, and `SHA256SUMS`.

After pushing the matching Git tag, attach the wheel and ZIP to that tag's pre-release.
