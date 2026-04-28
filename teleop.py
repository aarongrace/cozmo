# this is the stock code which I keep for reference
# unused
import sys
import time
import pycozmo
import pygame
from PIL import Image

# Driving settings
SPEED = 50.0  # mm/s

def on_camera_image(cli, image):
    """
    Callback function triggered when a new camera frame is received.
    Converts the PIL image to a Pygame surface and updates the display.
    """
    try:
        # Pycozmo provides a PIL image directly in the event
        mode = image.mode
        size = image.size
        data = image.tobytes()
        
        # Create Pygame image
        py_image = pygame.image.fromstring(data, size, mode)
        
        # Scale up for easier viewing (2x zoom)
        py_image = pygame.transform.scale(py_image, (640, 480))
        
        # Display on screen
        screen = pygame.display.get_surface()
        if screen:
            screen.blit(py_image, (0, 0))
            pygame.display.flip()
            
    except Exception as e:
        print(f"Error processing image: {e}")

def handle_input(cli):
    """
    Checks keyboard state and sends velocity commands.
    """
    keys = pygame.key.get_pressed()
    
    l_wheel = 0.0
    r_wheel = 0.0
    
    # WASD Controls
    # TODO: Edit l_wheel and r_wheel speed based on pressed keys!
    if keys[pygame.K_w]: # Forward
        pass
    elif keys[pygame.K_s]: # Backward
        pass
    if keys[pygame.K_a]: # Turn Left
        pass
    elif keys[pygame.K_d]: # Turn Right
        pass

    # Send drive command
    cli.drive_wheels(lwheel_speed=l_wheel, rwheel_speed=r_wheel, duration=0.2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Cozmo Teleop - WASD to Drive")
    with pycozmo.connect() as cli:
        # Enable camera
        cli.enable_camera(enable=True, color=True)
        
        # Register the camera callback
        # PyCozmo's client handles the decoding and fires this event
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

        running = True
        clock = pygame.time.Clock()

        while running:
            # Event handling (Exit)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Drive logic
            handle_input(cli)

            # Cap framerate to prevent flooding the network/CPU
            clock.tick(15)
            
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()