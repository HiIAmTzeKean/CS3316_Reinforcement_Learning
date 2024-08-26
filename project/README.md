# Requirements

The code can be run as specified. Examples will be provided in this section on
how to use the the program. There will be a need to install all packages before
the code can run. Note the following requirements after installing all prompted
packages:

- Python 3.11.19
- gymnasium[atari,accept-rom-license]
- gymnasium[mujoco]

To install the following requirements you can run the script below

```
pip install "gymnasium[atari,accept-rom-license]"
pip install "gymnasium[mujoco]"
```

# Usage

For Atari experiment specify the first argument as \code{atari} with the
following optional arguments. The environments that will work in this setup are
the following [VideoPinball-ramNoFrameskip-v4, BreakoutNoFrameskip-v4,
PongNoFrameskip-v4, BoxingNoFrameskip-v4] and the dueling argument is a boolean
specifier. Do note that you have to specify the argument as a capitalized
option.

```
python3.11 run.py --exp atari --env_name ENV --is_dueling BOOL
```

For Mujoco, the following arguments can also be specified. That is, only the
first option is required. Note that for the environments that can be specified,
the following environments can be tested on: [Hopper-v4, Humanoid-v4,
HalfCheetah-v4, Ant-v4] and the method option allows the user to choose between
[PPO, SAC] as the agent of choice.

```
python3.11 run.py --exp mujoco --env_name ENV --method METHOD
```