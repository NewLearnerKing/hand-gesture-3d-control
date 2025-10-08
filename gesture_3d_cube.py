import cv2
import mediapipe as mp
import math
import threading
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# ----------------- Hand Tracking -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
gesture_result = "Unknown"
scale_factor = 1.0
rotation = [0.0, 0.0]  # rotation_x, rotation_y
move_offset = [0.0, 0.0, 0.0]  # x, y, z

# ----------------- Hand Gesture Functions -----------------
def distance3D(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def is_finger_extended(hand, tip, mcp, hand_size):
    tip_lm = hand.landmark[tip]
    mcp_lm = hand.landmark[mcp]
    if (tip_lm.z - mcp_lm.z) < -0.15 * hand_size:
        return True
    if tip_lm.y < mcp_lm.y and abs(tip_lm.z - mcp_lm.z) < 0.1 * hand_size:
        return True
    return False

def get_gesture(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    hand_size = distance3D(wrist, middle_mcp)

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    pinch_dist = distance3D(thumb_tip, index_tip) / hand_size

    fingers_extended = 0
    fingers_curled = 0
    for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        if is_finger_extended(hand_landmarks, tip, mcp, hand_size):
            fingers_extended += 1
        else:
            fingers_curled += 1

    if pinch_dist < 0.35 and fingers_extended >= 2:
        return "Pinch"
    if fingers_curled >= 3:
        return "Fist"
    if fingers_extended >= 3:
        return "Open Palm"
    return "Unknown"

def hand_thread():
    global gesture_result, scale_factor, rotation, move_offset
    prev_distance = None
    prev_pinch_z = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            gestures = []
            hands_landmarks = results.multi_hand_landmarks

            # Detect gestures for all hands
            for hand_landmarks in hands_landmarks:
                gestures.append(get_gesture(hand_landmarks))

            # --- Double pinch for scaling ---
            if gestures.count("Pinch") == 2:
                hand1 = hands_landmarks[0].landmark[9]
                hand2 = hands_landmarks[1].landmark[9]
                curr_dist = distance3D(hand1, hand2)
                if prev_distance is not None:
                    scale_factor *= curr_dist / prev_distance
                    scale_factor = max(0.2, min(scale_factor, 3.0))
                prev_distance = curr_dist
                continue
            else:
                prev_distance = None

            # --- Fist + Open Palm rotation (3D) ---
            if "Fist" in gestures and "Open Palm" in gestures:
                for i, g in enumerate(gestures):
                    if g == "Open Palm":
                        hand_landmarks = hands_landmarks[i]
                        wrist = hand_landmarks.landmark[0]
                        index_mcp = hand_landmarks.landmark[5]
                        pinky_mcp = hand_landmarks.landmark[17]

                        # Vectors along the palm
                        v1 = [index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z]
                        v2 = [pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z]

                        # Cross product → palm normal
                        nx = v1[1]*v2[2] - v1[2]*v2[1]
                        ny = v1[2]*v2[0] - v1[0]*v2[2]
                        nz = v1[0]*v2[1] - v1[1]*v2[0]

                        # Convert normal to rotation angles
                        rotation[0] = math.degrees(math.atan2(ny, nz))  # pitch
                        rotation[1] = math.degrees(math.atan2(nx, nz))  # yaw


            # --- Single pinch for moving ---
            if "Pinch" in gestures:
                idx = gestures.index("Pinch")
                hand_landmarks = hands_landmarks[idx]
                cx = hand_landmarks.landmark[9].x - 0.5
                cy = 0.5 - hand_landmarks.landmark[9].y
                cz = hand_landmarks.landmark[9].z
                move_offset = [cx*5, cy*5, cz*10]  # store z offset
                prev_pinch_z = cz
            else:
                prev_pinch_z = None

        # Webcam preview (optional)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- OpenGL 3D Object -----------------
def draw_cube():
    glBegin(GL_QUADS)
    # Front face
    glColor3f(1,0,0)
    glVertex3f( 1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1,-1, 1)
    glVertex3f( 1,-1, 1)
    # Back face
    glColor3f(0,1,0)
    glVertex3f( 1, 1,-1)
    glVertex3f(-1, 1,-1)
    glVertex3f(-1,-1,-1)
    glVertex3f( 1,-1,-1)
    # Left face
    glColor3f(0,0,1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1,-1)
    glVertex3f(-1,-1,-1)
    glVertex3f(-1,-1, 1)
    # Right face
    glColor3f(1,1,0)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1,-1)
    glVertex3f(1,-1,-1)
    glVertex3f(1,-1, 1)
    # Top face
    glColor3f(1,0,1)
    glVertex3f( 1,1, 1)
    glVertex3f(-1,1, 1)
    glVertex3f(-1,1,-1)
    glVertex3f( 1,1,-1)
    # Bottom face
    glColor3f(0,1,1)
    glVertex3f( 1,-1, 1)
    glVertex3f(-1,-1, 1)
    glVertex3f(-1,-1,-1)
    glVertex3f( 1,-1,-1)
    glEnd()

def init_gl():
    glClearColor(0.2,0.2,0.2,1.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800/600, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def display():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(move_offset[0], move_offset[1], -10 + move_offset[2])  # include z movement
    glScalef(scale_factor, scale_factor, scale_factor)
    glRotatef(rotation[0], 1,0,0)
    glRotatef(rotation[1], 0,1,0)
    draw_cube()
    glutSwapBuffers()

def idle():
    glutPostRedisplay()

# ----------------- Main -----------------
threading.Thread(target=hand_thread, daemon=True).start()

glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
glutInitWindowSize(800,600)
glutCreateWindow("3D Cube Hand Control".encode())
init_gl()
glutDisplayFunc(display)
glutIdleFunc(idle)
glutMainLoop()
