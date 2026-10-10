"""Open the simulator window for manual or autonomous driving.

    .venv\Scripts\python run_sim.py            # goal seed 1000
    .venv\Scripts\python run_sim.py --seed 1003
    .venv\Scripts\python run_sim.py --driver baseline
"""

from robot_env.app import main

if __name__ == "__main__":
    main()
