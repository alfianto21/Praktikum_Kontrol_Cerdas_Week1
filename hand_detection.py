import cv2
import mediapipe
capture = cv2.VideoCapture(0)

mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands()

while True:
    success, frame = capture.read()
    imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tangan.process(imRGB)
    if results.multi_hand_landmarks:
        print("tangan")
    else: print("tidak ada")

    cv2.imshow("webcam", frame)
    cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break
capture.release()
cv2.destroyAllWindows()