from quadruped_pympc import config as cfg
from simulation import run_simulation
import pathlib

if __name__ == "__main__":
    # Set up output directory and file name
    output_dir = pathlib.Path("datasets/go1/flat_terrain")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "experiment1.h5"

    # Run simulation for 100 episodes, stacking all in the same file
    run_simulation(
        qpympc_cfg=cfg,
        num_episodes=100,
        render=True,
        recording_path=output_dir  # The simulation script will use this path and always write to 'test.h5'
    )

    ## to do: add noise to speed and get different types of traj
    ## ensure data is sufficient and rich