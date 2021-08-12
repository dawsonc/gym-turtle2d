import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces

from shapely.geometry import Polygon

from gym_turtle2d.systems.planar_lidar_system import Scene
from gym_turtle2d.systems.turtlebot_2d import TurtleBot2D


class Turtle2DEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # Create a scene based on that in 04c9
        obstacles = []
        obstacles.append(
            Polygon(
                [
                    (5.25, -5.25),
                    (5.25, -5.0),
                    (-5.25, -5.0),
                    (-5.25, -5.25),
                    (5.25, -5.25),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [(5.25, 5.0), (5.25, 5.25), (-5.25, 5.25), (-5.25, 5.0), (5.25, 5.0)]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (-5.0, -5.25),
                    (-5.0, 5.25),
                    (-5.25, 5.25),
                    (-5.25, -5.25),
                    (-5.0, -5.25),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [(5.25, -5.25), (5.25, 5.25), (5.0, 5.25), (5.0, -5.25), (5.25, -5.25)]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (-1.4605056102051046, 4.110354804605776),
                    (-2.251718543755679, 4.336664595534272),
                    (-2.6388829055549072, 2.983079983889303),
                    (-1.8476699720043324, 2.756770192960807),
                    (-1.4605056102051046, 4.110354804605776),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (1.0996938630562296, -1.8877761828352413),
                    (-0.11126041261783959, -3.132312574917047),
                    (0.7186360985718723, -3.9398154611212104),
                    (1.9295903742459415, -2.6952790690394046),
                    (1.0996938630562296, -1.8877761828352413),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (-1.3210542284862692, 3.302963377010861),
                    (-2.246470845205984, 4.095888249845055),
                    (-2.963579418115107, 3.2589562628457878),
                    (-2.038162801395393, 2.4660313900115938),
                    (-1.3210542284862692, 3.302963377010861),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (4.755251580487042, 1.8415029116222899),
                    (4.06094833335821, 2.5240346599292183),
                    (2.9040031561997495, 1.3471358632890116),
                    (3.598306403328581, 0.6646041149820832),
                    (4.755251580487042, 1.8415029116222899),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (2.170182506729752, -1.7491551467391564),
                    (1.3465929835069623, -1.6254130177586383),
                    (1.1934988866208032, -2.64436020495748),
                    (2.017088409843593, -2.7681023339379984),
                    (2.170182506729752, -1.7491551467391564),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (1.7076023573064951, -1.3835291747526826),
                    (2.414391935655882, -1.7418775016538055),
                    (3.0518463947989254, -0.48459196528526305),
                    (2.345056816449539, -0.12624363838414032),
                    (1.7076023573064951, -1.3835291747526826),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (0.5027429330025073, 2.4576546487952324),
                    (1.4644296074884937, 3.763868456650919),
                    (0.44835014432657294, 4.511946661049789),
                    (-0.5133365301594135, 3.205732853194103),
                    (0.5027429330025073, 2.4576546487952324),
                ]
            )
        )
        obstacles.append(
            Polygon(
                [
                    (0.6655320509884877, 0.8703707899849724),
                    (1.1576940290021827, 0.17722694372557113),
                    (2.1006616309947144, 0.8467745458632572),
                    (1.6084996529810196, 1.5399183921226582),
                    (0.6655320509884877, 0.8703707899849724),
                ]
            )
        )
        self.scene = Scene(obstacles)

        # Make a dynamics model and control period as well
        nominal_params = {"R": 0.0325, "L": 0.14}
        self.dynamics_model = TurtleBot2D(
            nominal_params,
            self.scene,
            dt=0.01,
            controller_dt=0.1,
            num_rays=32,
            field_of_view=(-np.pi, np.pi),
            max_distance=20.0,
        )
        self.dt = 0.1

        # Initialize the state
        self.state = torch.zeros(1, self.dynamics_model.n_dims)

        # Initialize the rendered
        self.fig = None
        self.ax = None

        # Create the action and observation space
        self.action_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(self.dynamics_model.n_controls,),
            dtype=np.float32,
        )
        num_obs = (
            self.dynamics_model.n_obs * self.dynamics_model.obs_dim
            + self.dynamics_model.n_dims
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32
        )

    def step(self, action):
        """Take a step given the state and action and return reward and cost"""
        # Action is a np array v, omega, which we convert to a tensor
        u = torch.tensor(action).reshape(1, self.dynamics_model.n_controls)

        # Update the state
        self.state = self.dynamics_model.zero_order_hold(self.state, u, self.dt)

        # Get the next observation
        lidar = self.dynamics_model.get_observations(self.state).reshape(-1)
        next_observation = torch.cat((lidar, self.state.reshape(-1))).numpy()

        # Get the reward based on distance to the goal
        distance_squared = (self.state[:, :2] ** 2).sum(dim=-1).reshape(-1, 1)
        # Phi is the angle from the current heading towards the origin
        angle_from_bot_to_origin = torch.atan2(-self.state[:, 1], -self.state[:, 0])
        theta = self.state[:, 2]
        phi = angle_from_bot_to_origin - theta
        # First, wrap the angle error into [-pi, pi]
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))
        phi = phi.reshape(-1, 1)
        reward = -(1.0 * distance_squared + 0.5 * (1 - torch.cos(phi))).item()

        # See if we're done
        goal_reached = (self.state[:, :2].norm(dim=-1) < 0.75).all()

        # See if we've hit anything and make that a cost
        in_collision = (self.scene.min_distance_to_obstacle(self.state) <= 0.2).any()
        # add a large cost if we're in collision
        if in_collision:
            cost = 100
        else:
            cost = 0.0

        info = {"cost_collision": cost, "cost": cost}

        print(info)

        # Return next_observation, reward, done, info
        return next_observation, reward, goal_reached, info

    def reset(self):
        # Get a random position that's not in collision
        in_collision = True
        while in_collision:
            self.state = torch.FloatTensor(1, self.dynamics_model.n_dims).uniform_(
                -4.5, 4.5
            )
            in_collision = (
                self.scene.min_distance_to_obstacle(self.state) <= 0.2
            ).any()

        # Get the first observation
        lidar = self.dynamics_model.get_observations(self.state).reshape(-1)
        obs = torch.cat((lidar, self.state.reshape(-1))).numpy()

        return obs

    def render(self, mode="human"):
        # Plot the environment and current state
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(1, 1)

        self.dynamics_model.plot_environment(self.ax)
        self.ax.plot(
            self.state[:, 0],
            self.state[:, 1],
            color="r",
            marker="o",
            markersize=12,
        )

        # Use pause instead of show
        plt.pause(self.dt)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
