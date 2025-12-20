import cv2
import numpy as np
import pygame
import random
import time
import mediapipe as mp

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 600, 400
SNAKE_RADIUS = 15  # Slightly larger for better image visibility
SPEED = 5        
START_DELAY = 3
FIST_THRESHOLD = 0.05

# ---------------- INIT ----------------
pygame.init()
pygame.mixer.init()

# Load Sounds
try:
    eat_sound = pygame.mixer.Sound("Snakegame-eat.mp3")
    game_over_sound = pygame.mixer.Sound("Snakegame-end.mp3")
except:
    print("Sound files not found, continuing without sound.")
    eat_sound = None

# --- NEW: LOAD AND SCALE IMAGES ---
try:
    grapes_img = pygame.image.load("grapes.png")
    broccoli_img = pygame.image.load("broccoli.png")
    # Scale images to match the snake size
    food_size = (SNAKE_RADIUS * 2, SNAKE_RADIUS * 2)
    grapes_img = pygame.transform.scale(grapes_img, food_size)
    broccoli_img = pygame.transform.scale(broccoli_img, food_size)
    food_types = [grapes_img, broccoli_img]
except Exception as e:
    print(f"Error loading images: {e}")
    food_types = None

clock = pygame.time.Clock()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ---------------- GAME DATA ----------------
snake = [[WIDTH//2, HEIGHT//2]]
score = 0
start_time = time.time()
head_pos = np.array([WIDTH//2, HEIGHT//2], dtype=float)

# NEW: Food data includes position AND type
food_pos = [random.randint(20, WIDTH-20), random.randint(20, HEIGHT-20)]
current_food_img = random.choice(food_types) if food_types else None

# ---------------- DRAW GAME ----------------
font_small = pygame.font.SysFont(None, 24)
font_big = pygame.font.SysFont(None, 72)

def draw_game(surface):
    surface.fill((20, 20, 20)) # Slightly lighter black
    
    # Draw Food
    if current_food_img:
        # Draw the image centered at food_pos
        img_rect = current_food_img.get_rect(center=(int(food_pos[0]), int(food_pos[1])))
        surface.blit(current_food_img, img_rect)
    else:
        # Fallback to red circle if images fail
        pygame.draw.circle(surface, (255, 0, 0), (int(food_pos[0]), int(food_pos[1])), SNAKE_RADIUS)

    # Draw Snake
    for i, segment in enumerate(snake):
        # Gradient effect: Head is brighter green than tail
        color = (0, 255, 0) if i == 0 else (0, 180, 0)
        pygame.draw.circle(surface, color, (int(segment[0]), int(segment[1])), SNAKE_RADIUS)
    
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
    if not ret: break
    frame = cv2.flip(frame, 1) # Flip webcam horizontally for intuitive control

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    target_pos = head_pos.copy()
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if not is_fist(hand):
            index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # Coordinates are now direct because we flipped the frame
            target_pos[0] = index_tip.x * WIDTH
            target_pos[1] = index_tip.y * HEIGHT

    # -------- GAME LOGIC --------
    game_surf = pygame.Surface((WIDTH, HEIGHT))
    if time.time() - start_time < START_DELAY:
        game_surf = draw_game(game_surf)
        num = font_big.render(str(START_DELAY - int(time.time() - start_time)), True, (255,255,255))
        game_surf.blit(num, (WIDTH//2 - 20, HEIGHT//2 - 30))
    else:
        direction = target_pos - head_pos
        distance = np.linalg.norm(direction)
        if distance > SPEED:
            direction = direction / distance * SPEED
            head_pos += direction

        snake.insert(0, head_pos.copy())
        if len(snake) > 15 + score*3:
            snake.pop()

        # Check food collision
        if np.linalg.norm(head_pos - np.array(food_pos)) < SNAKE_RADIUS*2:
            if eat_sound: eat_sound.play() 
            score += 1
            food_pos = [random.randint(20, WIDTH-20), random.randint(20, HEIGHT-20)]
            # Pick a new random image for the next food
            if food_types:
                current_food_img = random.choice(food_types)

        game_surf = draw_game(game_surf)

    # -------- DISPLAY --------
    game_img = pygame.surfarray.array3d(game_surf)
    game_img = cv2.transpose(game_img)
    game_img = cv2.cvtColor(game_img, cv2.COLOR_RGB2BGR)

    cam_h = HEIGHT
    cam_w = int(frame.shape[1]*cam_h/frame.shape[0])
    cam_resized = cv2.resize(frame, (cam_w, cam_h))

    combined = np.hstack((game_img, cam_resized))
    cv2.imshow("Hand Gesture Snake", combined)

    if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
        break

    clock.tick(60)

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
pygame.quit()