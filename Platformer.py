import pygame
import pygame_gui
import sys
import random
import cv2
import mediapipe as mp
import os
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow
from game_over import Ui_MainWindow  # Import giao diện từ file Python được sinh ra

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400
WHITE = (255, 255, 255)
FPS = 60
GRAVITY = 1
JUMP_HEIGHT = -20
PLAYER_WIDTH, PLAYER_HEIGHT = 50, 50
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 30, 30
GROUND_HEIGHT = 50

# PyGame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Mini Dinosaur - Made By PCĐ")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 50)

# Load assets
player_img = pygame.image.load("assets/player.png").convert_alpha()
player_img = pygame.transform.scale(player_img, (PLAYER_WIDTH, PLAYER_HEIGHT))
obstacle_img = pygame.image.load("assets/obstacle.png").convert_alpha()
obstacle_img = pygame.transform.scale(obstacle_img, (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
ground_img = pygame.image.load("assets/ground.png").convert_alpha()
ground_img = pygame.transform.scale(ground_img, (SCREEN_WIDTH, GROUND_HEIGHT))
background_img = pygame.image.load("assets/background.png").convert_alpha()
background_img = pygame.transform.scale(background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Mediapipe setup for hand tracking
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Game variables
player = pygame.Rect(50, 300, PLAYER_WIDTH, PLAYER_HEIGHT)
player_velocity = 0
is_jumping = False
obstacles = []
obstacle_speed = 5
SPAWN_OBSTACLE = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWN_OBSTACLE, 2500)
score = 0
fist_history = []

# High score functions
def check_high_score():
    try:
        with open("highscore.txt", "r") as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return 0

def save_high_score(new_high_score):
    with open("highscore.txt", "w") as file:
        file.write(str(new_high_score))

# GameOverWindow class
class GameOverWindow(QMainWindow):
    def __init__(self, score, high_score):
        super(GameOverWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Display scores
        self.ui.label_score.setText(f"Your Score: {score}")
        self.ui.label_high_score.setText(f"High Score: {high_score}")

        # Connect buttons
        self.ui.btn_try_again.clicked.connect(self.try_again)
        self.ui.btn_quit.clicked.connect(self.quit_game)

    def try_again(self):
        self.close()
        restart_game()

    def quit_game(self):
        pygame.quit()
        sys.exit()

# Hand detection
def detect_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    thumb_middle_distance = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5

    return thumb_index_distance < 0.05 and thumb_middle_distance < 0.05

# Restart the game
def restart_game():
    global player_velocity, is_jumping, obstacles, obstacle_speed, score, fist_history
    player_velocity = 0
    is_jumping = False
    obstacles = []
    obstacle_speed = 5
    score = 0
    fist_history = []
    main()

# Game over screen
def game_over():
    global high_score, score
    if score > high_score:
        high_score = score
        save_high_score(high_score)

    app = QApplication(sys.argv)
    game_over_window = GameOverWindow(score, high_score)
    game_over_window.show()
    app.exec_()
    restart_game()

# Draw the game
def draw():
    screen.blit(background_img, (0, 0))  # Draw background
    screen.blit(ground_img, (0, SCREEN_HEIGHT - GROUND_HEIGHT))  # Draw ground
    screen.blit(player_img, (player.x, player.y))  # Draw player

    # Draw obstacles
    for obstacle in obstacles:
        screen.blit(obstacle_img, (obstacle.x, obstacle.y))

    # Draw score
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()

# Main game loop
def main():
    global player_velocity, is_jumping, obstacles, obstacle_speed, score, fist_history

    cap = cv2.VideoCapture(0)  # Start webcam
    frame_skip = 0  # Skip frames to reduce computation

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)  # Flip webcam horizontally

            # Skip processing every 2nd frame
            if frame_skip % 2 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
            frame_skip += 1

            # Detect fist
            fist_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if detect_fist(hand_landmarks):
                        fist_detected = True
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Update fist detection history
            fist_history.append(fist_detected)
            if len(fist_history) > 5:
                fist_history.pop(0)

            # Jump if fist detected
            if sum(fist_history) >= 3 and not is_jumping:
                player_velocity = JUMP_HEIGHT
                is_jumping = True

            # Update player position
            player_velocity += GRAVITY
            player.y += player_velocity
            if player.y >= 300:
                player.y = 300
                is_jumping = False

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    sys.exit()
                if event.type == SPAWN_OBSTACLE:
                    obstacles.append(pygame.Rect(SCREEN_WIDTH, 300, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

            # Move obstacles and check collisions
            for obstacle in obstacles[:]:
                obstacle.x -= obstacle_speed
                if obstacle.x + OBSTACLE_WIDTH < 0:
                    obstacles.remove(obstacle)
                    score += 1

                if player.colliderect(obstacle):
                    cap.release()
                    game_over()
                    return

            draw()
            clock.tick(FPS)

            # Show webcam frame
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Start Menu
def start_menu():
    manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

    play_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 25), (200, 50)),
        text='Play Game',
        manager=manager
    )

    while True:
        time_delta = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == play_button:
                    return

            manager.process_events(event)

        screen.fill(WHITE)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()

# Run the game
high_score = check_high_score()
start_menu()
main()
