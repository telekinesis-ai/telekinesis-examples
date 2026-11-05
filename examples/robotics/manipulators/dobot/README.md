# Dobot: all 17 models in Rerun

Run from the telekinesis-examples repository root:

```powershell
conda activate synapse
python -m pip install --upgrade telekinesis-synapse xacro
git clone --depth 1 https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4.git .cache/DOBOT_6Axis_ROS2_V4
python examples/robotics/manipulators/dobot/prepare_dobot.py
python examples/robotics/manipulators/dobot/visualize_rerun.py
```

Skip cloning if the checkout exists. ROS is not required. The converter creates
17 URDFs and copies all 17 upstream SRDFs into `.cache/dobot`, records the source
commit, and validates mesh paths. It removes simulator plugins and clamps the
invalid Nova2S alpha value to 1. Meshes remain in the cloned repository;
regenerate if the checkout moves.

The gallery checks six joints, seven visual meshes, and finite transforms for
each model, saves `.cache/dobot/all_dobot.rrd`, and opens Rerun. Use `--headless`
to save only, `--save` for another recording destination, or `--description-dir`
for another prepared directory. Reopen a recording with:

```powershell
python -m rerun .cache/dobot/all_dobot.rrd
```

Supported models: CR3, CR3AF, CR5, CR5AF, CR6AL, CR7, CR10, CR10AF, CR12,
CR16, CR20, CR20AF, CR30H, ME6, Nova2, Nova2S, Nova5.

The gallery prefers the installed Synapse Dobot module. For wheels without it
(including the tested 0.10.8 wheel), `_models.py` supplies the same offline
wrappers using the installed native runtime. No sibling source checkout or
native rebuild is needed. Physical robot control is not implemented.
