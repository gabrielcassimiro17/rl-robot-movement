# Continuous Control with RL project
Repository for the Udacity RL Specialization second project Continuous Control using Actor-Critic methods.

# Project overview
This project has the objective to train an Agent using Actor-Critic methods to solve the Reacher environment.

## Enviroment & Task
The Reacher environment is a Unity-based simulation where an agent controls a double-jointed robotic arm to reach target locations. The state space is continuous, with 33 variables representing the arm's position, rotation, velocity, and angular velocities. The action space is also continuous, consisting of 4 variables for torque applied to the arm's joints, with each variable ranging from -1 to 1. The task is episodic, with the agent aiming to maximize its total reward over a fixed number of time steps. The environment is considered solved when the agent achieves an average score of 30 or more over 100 consecutive episodes.

<div align="center">
    <img src="reacher.gif">
</div>

# Usage

## Installing the environment

To install the env, select the environment that matches your operating system:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

The Github Repo already contains the environment for MacOS. If you are using another OS, you must download the environment and place it in the folder `rl-robot-movement/`.

## Training
To train the agent you must open the notebook `Continuous_Control_20.ipynb` and run all the cells. The agent will be trained and the weights will be saved in the file `actor_final.pth` and `critic_final.pth`.


## Visualizing trained agent
To visualize the trained agent you must open the notebook `Play.ipynb` and run all the cells.

# Dependencies
The dependencies are listed in the file `requirements.txt` in the folder `python/`. To install them you can run the following command:

```bash
cd python
pip install .
```

It is highly recommended to use a virtual environment to install the dependencies. you can do this by running the following commands:

	- Linux or Mac:
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- Windows:
	```bash
	conda create --name drlnd python=3.6
	activate drlnd
	```

# Files

```
.
├── Continuous_Control_20.ipynb -> Notebook to train the agent
├── LICENSE ---------------------> License file
├── Play.ipynb ------------------> Notebook to visualize the trained agent
├── README.md -------------------> This file
├── Report.md -------------------> Report of the project
├── actor_critic.py -------------> Actor-Critic model code
├── actor_final.pth -------------> Weights of the trained actor
├── basic_actor_critic_example_3.png -> Example of the Actor-Critic technique
├── critic_final.pth ------------> Weights of the trained critic
├── ddpg.py ---------------------> DDPG agent code
├── play.py ---------------------> Code to visualize the trained agent
├── reacher.gif -----------------> Gif of the environment
├── trained_agent.gif -----------> Gif of the trained agent
└── training_best.png -----------> Plot of the training scores
```


# References
- [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
- [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)
- [D4PG paper](https://arxiv.org/pdf/1804.08617.pdf)
