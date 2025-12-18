import cv2
import numpy as np
import pygame
import random
import time
import mediapipe as mp

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 600, 400
SNAKE_RADIUS = 10
SPEED = 5        # pixels per frame
START_DELAY = 3
FIST_THRESHOLD = 0.05

# ---------------- INIT ----------------
pygame.init()
clock = pygame.time.Clock()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- GAME DATA ----------------
snake = [[WIDTH//2, HEIGHT//2]]
food = [random.randint(20, WIDTH-20), random.randint(20, HEIGHT-20)]
score = 0
start_time = time.time()
head_pos = np.array([WIDTH//2, HEIGHT//2], dtype=float)

# ---------------- DRAW GAME ----------------
font_small = pygame.font.SysFont(None, 24)
font_big = pygame.font.SysFont(None, 72)

def draw_game(surface):
    surface.fill((0, 0, 0))
    for segment in snake:
        pygame.draw.circle(surface, (0, 255, 0), (int(segment[0]), int(segment[1])), SNAKE_RADIUS)
    pygame.draw.circle(surface, (255, 0, 0), (int(food[0]), int(food[1])), SNAKE_RADIUS)
    surface.blit(font_small.render(f"Score: {score}", True, (255,255,255)), (10,10))
    return surface

def is_fist(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    tips = [hand_landmarks.landmark[i] for i in [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]]
    avg_dist = np.mean([np.sqrt((wrist.x - t.x)**2 + (wrist.y - t.y)**2) for t in tips])
    return avg_dist < FIST_THRESHOLD

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    target_pos = head_pos.copy()
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if not is_fist(hand):
            index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # Invert X to match natural left-right movement
            target_pos[0] = (1 - index_tip.x) * WIDTH
            target_pos[1] = index_tip.y * HEIGHT

    # -------- COUNTDOWN --------
    game = pygame.Surface((WIDTH, HEIGHT))
    if time.time() - start_time < START_DELAY:
        game = draw_game(game)
        num = font_big.render(str(START_DELAY - int(time.time() - start_time)), True, (255,255,255))
        game.blit(num, (WIDTH//2 - 20, HEIGHT//2 - 30))
    else:
        # Move head smoothly towards fingertip
        direction = target_pos - head_pos
        distance = np.linalg.norm(direction)
        if distance > SPEED:
            direction = direction / distance * SPEED
        head_pos += direction

        # Add new head position
        snake.insert(0, head_pos.copy())
        if len(snake) > 20 + score*2:
            snake.pop()

        # Check food collision
        if np.linalg.norm(head_pos - np.array(food)) < SNAKE_RADIUS*2:
            score += 1
            food = [random.randint(20, WIDTH-20), random.randint(20, HEIGHT-20)]

        game = draw_game(game)

    # -------- DISPLAY --------
    game_img = pygame.surfarray.array3d(game)
    game_img = cv2.transpose(game_img)

    cam_h = HEIGHT
    cam_w = int(frame.shape[1]*cam_h/frame.shape[0])
    cam_resized = cv2.resize(frame, (cam_w, cam_h))

    combined = np.hstack((game_img, cam_resized))
    cv2.imshow("Snake Game | LEFT=Game RIGHT=Camera", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    clock.tick(60)

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("Game Over! Score:", score)
