import cv2
from ultralytics import YOLO

# 1. Load your trained model
model = YOLO("best_hand.pt") 

# 2. Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened")
    exit()

# Keypoint order expected by the trained hand pose model.
KEYPOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

NAME_TO_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}
SKELETON_CONNECTIONS = [
    ("WRIST", "THUMB_CMC"), ("THUMB_CMC", "THUMB_MCP"), ("THUMB_MCP", "THUMB_IP"), ("THUMB_IP", "THUMB_TIP"),
    ("WRIST", "INDEX_FINGER_MCP"), ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"), ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"), ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    ("WRIST", "MIDDLE_FINGER_MCP"), ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"), ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"), ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    ("WRIST", "RING_FINGER_MCP"), ("RING_FINGER_MCP", "RING_FINGER_PIP"), ("RING_FINGER_PIP", "RING_FINGER_DIP"), ("RING_FINGER_DIP", "RING_FINGER_TIP"),
    ("WRIST", "PINKY_MCP"), ("PINKY_MCP", "PINKY_PIP"), ("PINKY_PIP", "PINKY_DIP"), ("PINKY_DIP", "PINKY_TIP"),
    ("INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP"), ("MIDDLE_FINGER_MCP", "RING_FINGER_MCP"), ("RING_FINGER_MCP", "PINKY_MCP"),
]
SKELETON_INDEX_CONNECTIONS = [
    (NAME_TO_INDEX[start], NAME_TO_INDEX[end]) for start, end in SKELETON_CONNECTIONS
]

while True:
    success, frame = cap.read()
    
    if success:
        # 3. Inference
        results = model(frame, stream=True)

        for r in results:
            # Use the built-in plotter first (it usually draws skeletons automatically)
            annotated_frame = r.plot()

            # Optional: Manual drawing for more precision
            if r.keypoints is not None:
                # Get keypoints in x, y format
                # kpts shape is [num_objects, num_keypoints, 2]
                kpts = r.keypoints.xy.cpu().numpy()
                confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

                for hand_idx, hand in enumerate(kpts):
                    # Draw points
                    for kp_idx, kp in enumerate(hand):
                        if confs is not None and confs[hand_idx][kp_idx] < 0.25:
                            continue
                        cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

                    # Draw skeleton lines
                    for start_idx, end_idx in SKELETON_INDEX_CONNECTIONS:
                        if start_idx >= hand.shape[0] or end_idx >= hand.shape[0]:
                            continue
                        if confs is not None and (
                            confs[hand_idx][start_idx] < 0.25 or confs[hand_idx][end_idx] < 0.25
                        ):
                            continue
                        start_pt = (int(hand[start_idx][0]), int(hand[start_idx][1]))
                        end_pt = (int(hand[end_idx][0]), int(hand[end_idx][1]))
                        cv2.line(annotated_frame, start_pt, end_pt, (255, 0, 0), 2)

        # 4. Display
        cv2.imshow("YOLO Hand Skeleton", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
