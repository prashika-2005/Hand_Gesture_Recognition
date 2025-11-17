import pickle
import cv2
import mediapipe as mp
import numpy as np

# ---------- Load Trained Model ----------
print("🔹 Loading trained model...")
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
print("✅ Model loaded successfully!\n")

# ---------- Initialize MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ---------- Open Webcam ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam. Try changing camera index.")
    exit()

print("🎥 Webcam started! Show your gesture (A–J). Press 'ESC' to exit.\n")

# ---------- Label Dictionary ----------
labels_dict = {i: chr(65 + i) for i in range(10)}  # 0–9 → A–J

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Skipping empty frame...")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    # ---------- If Hands Detected ----------
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:  # 21 landmarks * 2 (x,y)
            data_aux = np.asarray(data_aux).reshape(1, -1)
            prediction = model.predict(data_aux)
            predicted_character = prediction[0]

            # Confidence (probability of prediction)
            prediction_proba = model.predict_proba(data_aux)
            confidence = np.max(prediction_proba) * 100

            # ---------- Display on Frame ----------
            cv2.putText(
                frame,
                f'{predicted_character} ({confidence:.2f}%)',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # ---------- Print in Console ----------
            print(f"👉 Predicted Gesture: {predicted_character} ({confidence:.2f}%)")

    cv2.imshow('Hand Gesture Recognition (A–J)', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        print("\n🛑 Exiting...")
        break

# ---------- Cleanup ----------
cap.release()
cv2.destroyAllWindows()
