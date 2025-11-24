# PA3_ECE_5463
PA3 Pick-and-Place Joint-Space Trajectory Planning & PID Control Homework Assignment ECE 5463

Overview:
This assignment requires you to derive the equations of motion, design a trajectory generator, and implement your own controller for a 2-DOF RR manipulator performing a pick-and-place task. 

Robot Description:
The robot is a 2-link planar arm. Joint 1 is revolute and fixed to ground. Joint 2 connects link 1 to link 2. End effector is at the end of Link 2.

Assume L1=0.5 m, L2=0.3 m, and joint limits: [-090, 90]

Pick-and-Place Configurations:
Home = [0°, 0°], Pick = [30°, -20°], Place = [45°, -30°] 

Required Tasks:
1. Must draw pint mass at the Pick and Place coordinates to show the robot is able to achieve position control. 
2. Must pause at the Pick and Place location, to demonstrate velocity control. 
3. Develop a smooth trajectory method of your choice.
4. Design, tune, and and justify your own controller (comment in the script), e_ss < 1 at the final configuration.
5. Simulate tracking and analyze results
6. Produce plots (end effector performance) at Pick and Place locations. This plot shows the behavior of the system.
7. Produce an animation showing the robot moving from Home position to Pick position, then moves to Place position. The must have appropriate speed to demonstrate the movements and stops. 
8. A short report explaining your strategy on solving the problem and developing the code. 
