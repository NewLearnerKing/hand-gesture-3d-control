import cv2
import mediapipe as mp
import math

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def distance3D(p1, p2):
    """3D Euclidean distance using x,y,z"""
    return math.sqrt((p1.x - p2.x) ** 2 +
                     (p1.y - p2.y) ** 2 +
                     (p1.z - p2.z) ** 2)

def is_finger_extended(hand, tip, mcp, hand_size):
    """Check if a finger is extended using depth + vertical check"""
    tip_lm = hand.landmark[tip]
    mcp_lm = hand.landmark[mcp]

    # If tip is much closer to camera than knuckle → pointing forward
    if (tip_lm.z - mcp_lm.z) < -0.15 * hand_size:
        return True

    # If tip is above knuckle (in y) and not much depth difference → pointing upward
    if tip_lm.y < mcp_lm.y and abs(tip_lm.z - mcp_lm.z) < 0.1 * hand_size:
        return True

    return False

def get_gesture(hand_landmarks):
    # Reference size (wrist to middle knuckle)
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    hand_size = distance3D(wrist, middle_mcp)

    # Thumb + index for pinch detection
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    pinch_dist = distance3D(thumb_tip, index_tip) / hand_size

    # Count extended & curled fingers
    fingers_extended = 0
    fingers_curled = 0
    for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        if is_finger_extended(hand_landmarks, tip, mcp, hand_size):
            fingers_extended += 1
        else:
            fingers_curled += 1

    # --- Pinch ---
    if pinch_dist < 0.35 and fingers_extended >= 2:
        return "Pinch"

    # --- Fist ---
    if fingers_curled >= 3:   # tolerant fist (majority curled)
        return "Fist"

    # --- Open Palm ---
    if fingers_extended >= 3:
        return "Open Palm"

    return "Unknown"

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)

            cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
