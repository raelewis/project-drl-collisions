"""
Author: Rachel Lewis
This is a helper file designed to generate a MuJoCo XML file for reinforcement learning testing.

Credits:
This file takes inspiration from DRL4's maze environment generation and utilizes techniques developed in their environments.
In particular, this file follows some direction of maze.py.
maze.py:            https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/maze/maze.py
"""

from enum import Enum
import numpy as np
import xml.etree.ElementTree as ET

class Hall(Enum):
    """
    This class enables the generation of each space of the environment.

    These options are supplied to Hallway._map in the form of a list of lists of characters. The characters determine
    what geoms, if any, should be created at a space, and can be expanded for future obstacles.

    Attributes
        WALL        ('W'):      This designates an edge, immovable wall that cannot be moved.
        CLEAR       ('C'):      This designates a free space where the agent can move safely.
        GOAL        ('G'):      This designates the end goal of the agent.
        START       ('S'):      This designates the starting position of the agent.
        OBSTACLE    ('O'):      This creates a door with a working hinge at the designated location.   
    """
    WALL = 'W'
    CLEAR = 'C'
    GOAL = 'G'
    START = 'S'
    OBSTACLE = 'O'

class Hallway():
    """
    This is a helper class for creating the MujoCo environment for collision avoidance.
    
    Attributes:
        _map:           A list of lists that contains the user supplied array of characters
        _width:         The width of the environment. This is the length of the first inner list of _map.
        _length:        The length of the environment. This is the length of the outer list of _map.
        _scale:         The size scale of the environment.
        _wall_height:   The wall height of the environment.
        _x_center:      The x center of the environment, determined by the width and scale of the environment.
        _y_center:      The y center of the environment, determined by the length and scale of the environment.
        _obstacles:     A list that holds each array space that is considered an obstacle.
        _empty_spaces:  A list that holds each space that is considered empty / free for the agent to move in.
        _walls:         A list that holds each wall, or immovable space, in the environment.
        _start:         The starting position of the agent.
        _goal:          The end goal position of the agent.
    """
    def __init__(self, scale, height, hallway_path=None):
        if hallway_path is None:
            hallway_path = [[]] # A default path is made if the user doesn't pass one
        self._map = hallway_path
        self._width = len(hallway_path[0]) # Size of inner array
        self._length = len(hallway_path) # Amount of arrays
        self._scale = scale
        self._wall_height = height

        # The way to keep the environment centered
        self._x_center = self.width / 2.0 * self.scale
        self._y_center = self.length / 2.0 * self.scale

        self._obstacles = []
        self._empty_spaces = []
        self._walls = []
        self._start = None
        self._goal = None

    """
    Property functions that serve as getter functions for Hallway.
    """
    @property
    def room_map(self):
        return self._map
    
    @property
    def width(self):
        return self._width
    
    @property
    def length(self):
        return self._length
    
    @property
    def obstacles(self):
        return self._obstacles
    
    @property
    def empty_spaces(self):
        return self._empty_spaces
    
    @property
    def walls(self):
        return self._walls
    
    @property
    def start(self):
        return self._start
    
    @property
    def goal(self):
        return self._goal
    
    @property
    def x_center(self):
        return self._x_center
    
    @property
    def y_center(self):
        return self._y_center
    
    @property
    def scale(self):
        return self._scale
    
    def _get_x_pos(self, x):
        """This method determines the x coordinate position of the space."""
        x_pos = (x + 0.5) * self.scale - self.x_center
        return x_pos
    
    def _get_y_pos(self, y):
        """This method determines the y coordinate position of the space."""
        y_pos = self.y_center - (y + 0.5) * self.scale
        return y_pos
    
    # Function that will build the path model xml structure
    @classmethod
    def _create_path(cls, room_map, scale, height, xml_path):
        """
        This function generates the simulation environment.
        
        Parameters:
            cls:        Python expression that denotes this as a class method.        
            room_map:   List of lists of characters that determines the environment layout.
            scale:      Determines the scale of the environment.
            height:     Determines the height of objects in the environment.
            xml_path:   The string name of the xml that contains the environment including camera, default lighting, etc.
        """
        # Retrieve the xml file data
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        # Create an instance of the hallway
        room = cls(scale, height, room_map)

        # Loop through the list of lists and generate the environment.
        for l in range(room._length):
            for w in range(room._width):
                # Convert arr[l][w] to a cell in the environment
                cell = room._map[l][w]
                x = room._get_x_pos(w)
                y = room._get_y_pos(l)

                # Switch statement to generate each cell of the environment
                match(cell):
                    case Hall.WALL.value:
                        room._create_wall(x, y, scale, height, worldbody)
                        room._walls.append(np.array([x, y])) # Save wall locations for future testing
                    case Hall.OBSTACLE.value:
                        room._create_door_obstacle(x, y, scale, height, tree)
                        room._obstacles.append(np.array([x, y])) # Save obstacle locations for future testing
                    case Hall.CLEAR.value:
                        room._empty_spaces.append(np.array([x, y])) # Save empty / free move spaces for future testing
                    case Hall.GOAL.value:
                        if room.goal is None:
                            room._goal = np.array([x, y]) # Save the goal position for reward calculation
                            room._create_goal_visual(x, y, scale, height, worldbody) # Temporary function to add a visualization of the goal
                        else:
                            print("Error: multiple goals provided by user.")
                            print("Exiting...")
                            return -1
                    case Hall.START.value:
                        if room.start is None:
                            room._start = np.array([x, y]) # Save start position for the agent
                            room._add_test_agent(x, y, scale, height, worldbody, tree.find(".//actuator")) # Temporary function to add a visual agent
                        else:    
                            print("Error: multiple start positions provided by user.")
                            print("Exiting...")
                            return -1
                    case _:
                        print("Unknown option in room configuration.")
                        print("Exiting...")
                        return -1
        
        # Write the xml tree to a new xml file and return the path of that file
        env_xml_path = xml_path[:]
        env_xml_path = env_xml_path.replace("default_env.xml", "env.xml")
        tree.write(env_xml_path)

        return room, env_xml_path

    def _create_wall(self, x, y, scale, height, worldbody):
        """
        This function adds a wall / immovable object to the xml tree.
        
        Parameters:
            x:          The x coordinate of the wall
            y:          The y coordinate of the wall
            scale:      The scale of the environment
            height:     The height of the environment
            worldbody:  The parent node of the wall
        """
        ET.SubElement(worldbody, 
                      "geom",
                      type="box",
                      name=f"wall_{x}_{y}",
                      pos=f"{x} {y} {height / 2 * scale}",
                      size=f"{0.5 * scale} {0.5 * scale} {height / 2 * scale}",
                      rgba="1 1 1 1",
                      contype="1")
        

    def _create_door_obstacle(self, x, y, scale, height, tree):
        """
        This function adds a door obstacle to the xml tree.
        
        Parameters:
            x:          The x coordinate of the wall
            y:          The y coordinate of the wall
            scale:      The scale of the environment
            height:     The height of the environment
            tree:       The variable containing the tree of the xml file
        """

        door_num = len(self._obstacles)
        worldbody = tree.find(".//worldbody")
        door_body = ET.SubElement(worldbody,
                                    "body",
                                    name=f"door_body_{door_num}",
                                    pos=f"{x} {y+scale/2} {height / 2 * scale + .05}")
        # Create the door hinge
        ET.SubElement(door_body,
                        "joint",
                        name=f"door_hinge_{door_num}",
                        pos=f"{scale/2} {0} {(height / 2 * scale) / 2}",
                        axis="0 0 1",
                        range="0 -90")
        # Create the door shape
        ET.SubElement(door_body,
                        "geom",
                        name=f"door_{door_num}",
                        type="box",
                        size=f"{scale/2 - .1} {.1 * scale} {height / 2 * scale - .05}",
                        rgba="0.4 0.24 0.0 1",
                        contype="1")
        # Add an actuator for the door hinge
        act = tree.find(".//actuator")
        ET.SubElement(act,
                      "motor",
                      name=f"door_{door_num}_hinge",
                      joint=f"door_hinge_{door_num}")

    def _create_goal_visual(self, x, y, scale, height, worldbody):
        """
        This function adds a visualization of the goal to the xml tree.
        
        Parameters:
            x:          The x coordinate of the wall
            y:          The y coordinate of the wall
            scale:      The scale of the environment
            height:     The height of the environment
            worldbody:  The parent node of the goal
        """
        """
        def update_target_site_pos(self):
            self.point_env.model.site_pos[self.target_site_id] = np.append(
                self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
            )"""
        body = ET.SubElement(worldbody,
                             "body",
                             name=f"goal")
        ET.SubElement(body,
                      "light",
                      directional="false",
                      pos=f"{x} {y} {height}",
                      ambient="1 0 1")
        ET.SubElement(body,
                      "site",
                      name="target",
                      pos=f"{x} {y} {height / 2 * scale}",
                      size=f"{0.2 * scale}",
                      rgba="1 0 0 0.7",
                      type="sphere",)
        
    def _add_test_agent(self, x, y, scale, height, worldbody, act):
        """
        This function adds a simple test "agent" to the xml tree.
        
        Parameters:
            x:          The x coordinate of the wall
            y:          The y coordinate of the wall
            scale:      The scale of the environment
            height:     The height of the environment
            worldbody:  The parent node of the agent
        """
        body = ET.SubElement(worldbody,
                             "body",
                             name="test-agent",
                             pos = f"0 0 {height / 2 * scale}")

        # Add the 'agent'
        ET.SubElement(body,
                    "geom",
                    name="agent",
                    type="sphere",
                    size=f"{0.2 * scale}",
                    rgba="0 0 1 1",
                    contype="1",
                    condim="4")
        
        # Make this sphere a free joint
        ET.SubElement(body,
                    "joint",
                    type="slide",
                    axis="1 0 0",
                    name="agent_joint_x")
        ET.SubElement(body,
                    "joint",
                    type="slide",
                    axis="0 1 0",
                    name="agent_joint_y")
        
        ET.SubElement(act,
                      "motor",
                      name=f"agent_motor_x",
                      ctrlrange="-1.0 1.0",
                      joint=f"agent_joint_x",
                      ctrllimited="true",
                      gear= "100")
        
        ET.SubElement(act,
                      "motor",
                      name=f"agent_motor_y",
                      ctrlrange="-1.0 1.0",
                      joint=f"agent_joint_y",
                      ctrllimited="true",
                      gear= "100")