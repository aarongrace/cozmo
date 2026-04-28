import pycozmo
import pygame
import time
import math
import random

# --- CONFIGURATION ---
WINDOW_SIZE = 800       # 800x800 pixel window
CENTER_X = WINDOW_SIZE // 2
CENTER_Y = WINDOW_SIZE // 2
SPEED = pycozmo.robot.MAX_WHEEL_SPEED.mmps / 2

# Global State
robot_pose = {'x': 0, 'y': 0, 'angle': 0, 'is_bumped': False, 'is_cliff': False}

behavior_states = {"Collision", "Avoid", "Cliff", "Wander"}

goal_angle = None
should_back_up = False

def world_to_screen(x, y):
    """ Converts robot mm coordinates to pygame screen coordinates"""
    # TODO: Convert x, y robot pose to location on screen
    # Return screen_x, screen_y

    # TODO update return
    return 0, 0

def on_robot_state(cli, pkt: pycozmo.protocol_encoder.RobotState):
    """ Updates the robot's internal position 30 times a second """
    global robot_pose
    robot_pose['x'] = pkt.pose_x
    robot_pose['y'] = pkt.pose_y
    robot_pose['angle'] = pkt.pose_angle_rad
    
    # Check cliff sensor
    if pkt.cliff_data_raw[0] < 200: 
        robot_pose['is_cliff'] = True
    else:
        robot_pose['is_cliff'] = False

def on_bump(cli, pkt):
    """ Triggered when accelerometer spikes """
    global robot_pose
    robot_pose['is_bumped'] = True
    print("Bumped!")
    print("Bumped!")
    print("Bumped!")
    print("Bumped!")
    print("Bumped!")
    print("Bumped!")
    print("Bumped!")


def run_lab(cli:pycozmo.client.Client):    
    global should_back_up
    global goal_angle
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Cozmo Mapper")
    screen.fill((255, 255, 255)) # Start with White (Unknown) background
    
    # setup listeners for state 
    cli.add_handler(pycozmo.protocol_encoder.RobotState, on_robot_state)
    cli.add_handler(pycozmo.protocol_encoder.RobotPoked, on_bump)
    
    # main loop
    running = True
    state = "Wander"
    
    while running:
        print(f"Current State: {state}")

        # Stop on window exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                cli.stop_all_motors()

        # Get position for screen
        sx, sy = world_to_screen(robot_pose['x'], robot_pose['y'])
        
        if state == "Collision":
            # Stop motors
            cli.stop_all_motors()
            print("Collision! Marking Obstacle.")
            
            # Draw BLACK rectangle at current location
            rect = pygame.Rect(sx-10, sy-10, 20, 20)
            pygame.draw.rect(screen, (0, 0, 0), rect)


            goal_angle = robot_pose['angle'] * random.random() + math.pi / 2
            should_back_up = True

            robot_pose['is_bumped'] = False
            state = "Avoid"


        elif state == "Cliff":
            print("Cliff! Marking Obstacle.")
            # Stop motors
            cli.stop_all_motors()
            
            # Draw Red rectangle at current location
            rect = pygame.Rect(sx-10, sy-10, 20, 20)
            pygame.draw.rect(screen, (255, 0, 0), rect)
            
            goal_angle = robot_pose['angle'] * random.random() + math.pi / 2
            state = "Avoid"
            

        elif state == "Avoid":
            if should_back_up:
                print("Backing Up")
                cli.drive_wheels(lwheel_speed=-SPEED, rwheel_speed=-SPEED, duration=0.5)
                time.sleep(0.6)  # wait for backing up to finish
                should_back_up = False

            if goal_angle is not None:
                print(f"Turning to {goal_angle} degrees")
                cli.drive_wheels(lwheel_speed=SPEED, rwheel_speed=-SPEED, duration=1.3)
                time.sleep(1.3)
                # new_pose = pycozmo.util.Pose(x=robot_pose['x'], y=robot_pose['y'], angle_z=goal_angle)
                # cli.go_to_pose(new_pose).wait_for_completed()
        
            state="Wander"
            
            
        elif state == "Wander":
            # Draw green trail at current location (space is free of obstacles)
            pygame.draw.circle(screen, (0, 255, 0), (sx, sy), 5) 
            
            if robot_pose['is_bumped']:
                state = "Collision"
            elif robot_pose['is_cliff']:
                state = "Cliff"
            else:
                cli.drive_wheels(lwheel_speed=SPEED, rwheel_speed=SPEED, duration=0.1)
                # time.sleep(0.1)
                
        else:
            print(f"Unknown State: {state}")

        # Update pygame screen
        pygame.display.flip()


    pygame.quit()

# Connect to robot
with pycozmo.connect() as cli:
    cli.set_head_angle(0.0)
    cli.set_lift_height(0.0)
    # Enable camera/sensors to ensure data stream works
    cli.enable_camera(True)
    run_lab(cli)