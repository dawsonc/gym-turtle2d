import gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces

from shapely.geometry import Polygon

from gym_turtle2d.systems.planar_lidar_system import Scene
from gym_turtle2d.systems.turtlebot_2d import TurtleBot2D


def get_training_obstacles():
    obstacles = []
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
    return obstacles


def get_selected_obstacles():
    obstacles = []
    obstacles.append(
        Polygon(
            [
                (-4.2136676463868366, 0.5679646499153548),
                (-2.743330656989159, -0.30710755158835323),
                (-2.0696517186228496, 0.8248391156192505),
                (-3.5399887080205277, 1.6999113171229587),
                (-4.2136676463868366, 0.5679646499153548),
            ]
        )
    )
    obstacles.append(
        Polygon(
            [
                (0.2810067281373088, 1.8313846995884488),
                (-1.1178221533074106, 1.1851621034840258),
                (-0.46066806204718935, -0.2373293537255825),
                (0.9381608193975304, 0.4088932423788404),
                (0.2810067281373088, 1.8313846995884488),
            ]
        )
    )
    obstacles.append(
        Polygon(
            [
                (3.0150536444465867, -3.6712772807033898),
                (4.4733962529696205, -3.4652282279839715),
                (4.31357901681553, -2.3340981306412214),
                (2.8552364082924964, -2.54014718336064),
                (3.0150536444465867, -3.6712772807033898),
            ]
        )
    )
    obstacles.append(
        Polygon(
            [
                (1.9915836749108597, 3.696365684856654),
                (1.1906921081176052, 4.626324483349467),
                (0.05418969918334149, 3.6475551051459822),
                (0.8550812659765963, 2.7175963066531694),
                (1.9915836749108597, 3.696365684856654),
            ]
        )
    )
    obstacles.append(
        Polygon(
            [
                (-0.8220082932121368, 1.2901466027527497),
                (-2.0028796561132642, 0.6294746466154549),
                (-1.2854031624122628, -0.6529279695327568),
                (-0.10453179951113567, 0.007743986604538122),
                (-0.8220082932121368, 1.2901466027527497),
            ]
        )
    )
    obstacles.append(
        Polygon(
            [
                (3.607829125311771, 0.5473325823272353),
                (4.215626943185372, 2.0756404349101327),
                (2.902897580988146, 2.597704128068316),
                (2.295099763114544, 1.0693962754854187),
                (3.607829125311771, 0.5473325823272353),
            ]
        )
    )
    obstacles.append(
        Polygon(
            [
                (-1.7080890714472434, -1.985992050420569),
                (-0.5808673151891823, -1.7964378971591917),
                (-0.7793064920166237, -0.616379567836322),
                (-1.9065282482746848, -0.8059337210976993),
                (-1.7080890714472434, -1.985992050420569),
            ]
        )
    )
    obstacles.append(
        Polygon(
            [
                (3.0496394871572035, -1.6500303672785277),
                (2.1257656941885603, -3.1212508648447477),
                (3.3240219557738753, -3.873712891556976),
                (4.247895748742518, -2.402492393990756),
                (3.0496394871572035, -1.6500303672785277),
            ]
        )
    )
    return obstacles


class Turtle2DEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    use_training_scene = False
    use_selected_scene = True

    def __init__(self):
        # Create a scene based on that in 04c9 or randomly
        obstacles = []
        # add walls
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

        if self.use_training_scene:
            obstacles += get_training_obstacles()
            self.scene = Scene(obstacles)
        elif self.use_selected_scene:
            obstacles += get_selected_obstacles()
            self.scene = Scene(obstacles)
        else:
            self.scene = Scene(obstacles)
            num_obstacles = 8
            box_size_range = (0.75, 1.75)
            position_range = (-4.0, 4.0)
            rotation_range = (-np.pi, np.pi)
            self.scene.add_random_boxes(
                num_obstacles,
                box_size_range,
                position_range,
                position_range,
                rotation_range,
            )

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
        self.u_low = torch.tensor([[0.1, -3 * np.pi]])
        self.u_high = torch.tensor([[2.0, 3 * np.pi]])
        self.action_space = spaces.Box(
            low=self.u_low.squeeze().numpy(),
            high=self.u_high.squeeze().numpy(),
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

        # Scale by the semi-range and offset to the center
        u_semirange = (self.u_high - self.u_low) / 2.0
        u_center = (self.u_high + self.u_low) / 2.0
        u = u * u_semirange + u_center

        # Clip the action to the limits
        u = torch.clip(u, self.u_low, self.u_high)

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

        # See if we're done (either reached the goal and get a bonus or hit something
        # and incur a cost)
        goal_reached = (self.state[:, :2].norm(dim=-1) < 0.75).all()
        if goal_reached:
            reward += 1000.0

        # See if we've hit anything and make that a cost
        min_distance_to_obstacle = self.scene.min_distance_to_obstacle(self.state)
        in_collision = (min_distance_to_obstacle <= 0.0).any()
        # add a large cost if we're in collision
        cost = 1000 * F.relu(0.2 - min_distance_to_obstacle).item()

        done = goal_reached or in_collision
        done = done.item()

        info = {
            "cost_collision": cost,
            "cost": cost,
            "goal_reached": goal_reached.item(),
            "in_collision": in_collision.item(),
        }

        # Return next_observation, reward, done, info
        return next_observation, reward, done, info

    def reset(self):
        # If we're not using the training scene, randomize the environment
        room_size = 10.0
        num_obstacles = 8
        box_size_range = (0.75, 1.75)
        position_range = (-4.0, 4.0)
        rotation_range = (-np.pi, np.pi)
        self.scene = Scene([])
        self.scene.add_walls(room_size)
        if self.use_training_scene:
            self.scene.obstacles += get_training_obstacles()
        elif self.use_selected_scene:
            self.scene.obstacles += get_selected_obstacles()
        else:
            self.scene.add_random_boxes(
                num_obstacles,
                box_size_range,
                position_range,
                position_range,
                rotation_range,
            )
        self.dynamics_model.scene = self.scene

        # Get a random position that's not in collision
        in_collision = True
        while in_collision:
            self.state = torch.FloatTensor(1, self.dynamics_model.n_dims).uniform_(
                -4.5, 4.5
            )
            if self.use_selected_scene:
                self.state = torch.tensor([[-4.0, 4.0, 0.0]])
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

        self.ax.clear()

        self.dynamics_model.plot_environment(self.ax)
        self.ax.arrow(
            self.state[:, 0],
            self.state[:, 1],
            0.4 * np.cos(self.state[:, 2]),
            0.4 * np.sin(self.state[:, 2]),
        )

        self.ax.plot(
            self.state[:, 0],
            self.state[:, 1],
            color="r",
            marker="o",
            markersize=8,
        )

        # Use pause instead of show
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
