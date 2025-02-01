import pyautogui
import pytesseract
from PIL import Image
import cv2
import numpy as np
import time
import pdb; 
import matplotlib.pyplot as plt

class NestOfThornsAI:
    def __init__(self):
        self.treasure_count = 0
        self.skill_count = 0
        self.running = True
        self.attempt_number = 0

        # Set Tesseract path if needed (adjust based on your setup)
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    def start_game(self):
        """
        Automates starting the game by detecting text dynamically and clicking buttons when they appear.
        """

        self.attempt_number += 1

        # Detect and click each required button based on text
        self.wait_and_click_text(800,950, 'PLAY')
        self.wait_and_click_text(500,600, 'DRAGONUS')
        self.wait_and_click_text(800,950, 'NEXT')
        self.wait_and_click_text(800,950, 'START')

        self.wait_for_the_game_to_start()
        print('Game has started!')

        directions = self.detect_treasures_and_direction()

        if directions[0] == directions[1]:
            print(f"Treasures are in the same direction: {directions[0]}")
        else:
            print(f"Treasures are not in the same direction: {directions}")
            self.restart()




    def restart(self):
        print('Unlucky: Restarting the game!')
        pyautogui.press('esc')
        time.sleep(0.5)
        self.wait_and_click_text(852, 631, 'Leave Game')
        time.sleep(0.5)
        self.start_game()

    def wait_and_click_text(self, x, y, target_text):
        """
        Continuously captures the screen and waits for the specified text to appear, then clicks it.
        
        :param target_text: The text to search for on the screen (e.g., 'PLAY', 'NEXT').
        """
        pyautogui.click(x, y)
        print(f"Clicked on '{target_text}' at ({x}, {y}).")
        time.sleep(0.5)  # Give time for the game to load the next screen

    def wait_for_the_game_to_start(self):
        while True:
            screen_capture = pyautogui.screenshot(region=(800,100,200,100))
            screen_capture_np = np.array(screen_capture)

            if self.find_text_in_image(screen_capture_np, '7'):
                return
            else:
                pyautogui.click(500, 500)

            time.sleep(0.1)

    def find_text_in_image(self, image, target_text):
        """
        Detects and locates specified text in the given image using OCR and visualizes it.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)

        for i in range(len(data['text'])):
            detected_text = data['text'][i].strip().lower()
            print(detected_text)
            if target_text.lower() in detected_text:
                return True

        return False

    def detect_treasures_and_direction(self):
        """
        Detects treasures on the minimap and determines if they are in the same direction.

        :param minimap_image_path: Path to the minimap image.
        :return: The direction if treasures are aligned, otherwise False.
        """
        # Load the minimap image
        minimap = pyautogui.screenshot(region=(15,770,260,260))
        minimap = np.array(minimap)
        minimap = cv2.resize(minimap, (520, 520), interpolation=cv2.INTER_LINEAR)
        treasure_positions = self.detect_treasures_with_color_filtering(minimap)

        if len(treasure_positions) != 2:
            print("Could not detect exactly two treasures.")
            return False

        # Determine if treasures are in the same direction
        directions = self.evaluate_treasure_direction(treasure_positions)

        self.save_minimap(treasure_positions, minimap, directions)

        return directions


    def detect_treasures_with_color_filtering(self, minimap):
        # Load the minimap image
        minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_RGB2HSV)  # Use RGB2HSV for screenshots
        # pdb.set_trace()

        # Define HSV range for golden/yellow treasure color
        lower_gold = np.array([12, 150, 15])   # Adjust as needed
        upper_gold = np.array([15, 250, 150]) # Adjust as needed

        # Create a mask to isolate the golden regions
        mask = cv2.inRange(minimap_hsv, lower_gold, upper_gold)

        # Find contours in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the center positions of detected treasures
        treasure_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 120:  # Filter out small noise
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    treasure_positions.append((cX, cY))

        # self.show_minimap(treasure_positions, minimap)

        return treasure_positions


    def show_minimap(self, treasure_positions, minimap):
        # Draw circles around detected treasures for visualization
        for (x, y) in treasure_positions:
            cv2.circle(minimap, (x, y), 10, (0, 255, 0), 2)

        # Display the minimap with detected treasures
        plt.figure(figsize=(6, 6))
        plt.imshow(minimap)
        plt.title("Detected Treasures (Color Filtering)")
        plt.axis('off')
        plt.show()

    def save_minimap(self, treasure_positions, minimap, detected_directions):
        """
        Draws detected treasures on the minimap and saves it as 'direction.png'.
        
        :param treasure_positions: List of (x, y) coordinates of detected treasures.
        :param minimap: The minimap image (NumPy array).
        :param detected_directions: (Optional) Directions to annotate on the minimap.
        """
        
        # Draw circles around detected treasures for visualization
        for (x, y) in treasure_positions:
            cv2.circle(minimap, (x, y), 10, (0, 255, 0), 2)  # Green circles for treasures

        # Optionally, annotate detected directions
        cv2.putText(minimap, f"Direction: {detected_directions}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert from RGB to BGR before saving if using PyAutoGUI screenshot
        minimap_bgr = cv2.cvtColor(minimap, cv2.COLOR_RGB2BGR)
        
        # Save the minimap with detected treasures
        cv2.imwrite(f'attempt{self.attempt_number}.png', minimap_bgr)
        print(f"Minimap saved as 'attempt{self.attempt_number}.png'.")

    def evaluate_treasure_direction(self, treasure_positions):
        """
        Evaluates the direction of treasures relative to the player at the center.

        :param treasure_positions: List of (x, y) positions of treasures.
        :return: Direction if aligned, otherwise False.
        """
        center_x, center_y = 260, 260  # Assuming the center based on minimap size (adjust if needed)

        # Calculate angles from center to each treasure
        angles = []
        for (x, y) in treasure_positions:
            dx = x - center_x
            dy = center_y - y  # Invert y-axis for traditional angle measurement
            angle = np.degrees(np.arctan2(dy, dx))
            angles.append(angle)

        # Normalize angles to sectors (e.g., N, NE, E, SE, S, SW, W, NW)
        directions = [self.angle_to_direction(angle) for angle in angles]

        return directions


    def angle_to_direction(self, angle):
        """
        Converts an angle to a compass direction.

        :param angle: Angle in degrees.
        :return: Compass direction as a string.
        """
        if -22.5 <= angle < 22.5:
            return 'E'
        elif 22.5 <= angle < 67.5:
            return 'NE'
        elif 67.5 <= angle < 112.5:
            return 'N'
        elif 112.5 <= angle < 157.5:
            return 'NW'
        elif (157.5 <= angle <= 180) or (-180 <= angle < -157.5):
            return 'W'
        elif -157.5 <= angle < -112.5:
            return 'SW'
        elif -112.5 <= angle < -67.5:
            return 'S'
        elif -67.5 <= angle < -22.5:
            return 'SE'
        else:
            return 'Unknown'

    def run(self):
        """
        Main loop controlling the AI.
        """
        self.start_game()

        while self.running:
            # Add further logic for in-game interactions here
            pass

# Start AI
ai = NestOfThornsAI()
print("Starting the game in 2 seconds... Switch to your game window now!")
time.sleep(2)
ai.run()

