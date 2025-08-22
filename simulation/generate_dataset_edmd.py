from quadruped_pympc import config as cfg
from simulation import run_simulation
import pathlib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDMD dataset for quadruped.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../datasets/go1/flat_terrain",
        help="Directory to save the output dataset (default: datasets/go1/flat_terrain)"
    )
    parser.add_argument(
        "--vel_type",
        type=str,
        default="forward+rotate",
        choices=["forward", "random", "forward+rotate", "human"],
        help="Type of base velocity command (default: forward+rotate)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to generate (default: 1)"
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_simulation(
        qpympc_cfg=cfg,
        num_episodes=args.episodes,
        render=True,
        recording_path=output_dir,
        recording_filename="experimentxx.h5", 
        base_vel_command_type=args.vel_type
    )

    ## to do: add noise to speed and get different types of traj
    ## ensure data is sufficient and rich