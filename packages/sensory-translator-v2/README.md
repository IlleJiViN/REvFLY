# REvFLY Python SDK and CLI

Python SDK and CLI for translating joint trajectories into continuous motor rates and replaying them with FlyMimic MuJoCo. The wheel installs the engine and its APIs. The release ZIP supplies the checkpoints, physics assets, and optional local Control Studio.

## Install from GitHub Releases

Download the ZIP from [the release page](https://github.com/IlleJiViN/REvFLY/releases/tag/v2.0.0a7). It includes the wheel and the runtime bundle. Extract the wheel into the same directory as the ZIP before installing, or download it separately. With GitHub CLI configured, you can download both files together.

```bash
gh release download v2.0.0a7 --repo IlleJiViN/REvFLY \
  --pattern 'sensory_translator_v2-2.0.0a7-*.whl' \
  --pattern 'revfly-2.0.0a7-local-beta.zip'
```

The physics workflow was tested on Linux x86-64 with Python 3.11. On Linux, you can install CPU Torch before the wheel to avoid pulling CUDA dependencies. Rendering uses the host's OpenGL/EGL libraries.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu
python -m pip install './sensory_translator_v2-2.0.0a7-py3-none-any.whl[physics]'
revfly install-assets ./revfly-2.0.0a7-local-beta.zip
revfly verify-assets
```

Assets are stored in `~/.local/share/sensory-translator-v2/bundle` or under `XDG_DATA_HOME`. An explicit `--asset-root` or `SENSORY_TRANSLATOR_V2_ASSET_ROOT` selects a different bundle. Source checkouts with the legacy physics assets keep their checkout defaults. Add `--replace` to install a different verified bundle in place. An extracted `bundle/` directory is also a valid input.

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

from sensory_translator_v2.engine import MultiJointEngine

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
python -m pip install 'sensory-translator-v2[physics] @ git+https://github.com/IlleJiViN/REvFLY.git@v2.0.0a7#subdirectory=packages/sensory-translator-v2'
```

## Build and publish a team release

In a checkout that contains the trained checkpoints and FlyMimic assets, build the wheel and export the runtime files:

```bash
python -m pip wheel --no-deps --wheel-dir dist ./packages/sensory-translator-v2
revfly export-assets --asset-root . --dest dist/bundle --include-v2-models --include-ui
```

After editing UI sources, build `apps/control_ui` before exporting. The beta ZIP contains the wheel, `bundle/`, this README, third-party notices, the upstream FlyMimic license, and `SHA256SUMS`. Use wheel assets for the GitHub release. The setuptools source tarball currently omits the package modules because the V2 build project uses the repository's shared `src/` tree.

After pushing the matching Git tag, attach the wheel and ZIP to that tag's pre-release.
