import argparse

pre = argparse.ArgumentParser()
pre.add_argument('--exp', required=True, type=str,
                 choices=['atari', 'mujoco'],
                 help='experiment to run')
args_pre, extra = pre.parse_known_args()

if args_pre.exp == 'atari':
    import atari_exp.main
    print("Atari Done")
elif args_pre.exp == 'mujoco':
    import mujoco_exp.main
    print("Mujoco Done")
else:
    print("Invalid experiment")