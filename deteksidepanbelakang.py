import cv2
import mediapipe
capture = cv2.VideoCapture(0)
mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands(max_num_hands=1)
mpdraw = mediapipe.solutions.drawing_utils
while True:
    success, img = capture.read()
    if not success:
        print("Webcam tidak terbaca")
        break
    # mirror webcam
    image = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:
        for titiktangan, hand in zip(results.multi_hand_landmarks,
                                      results.multi_handedness):
            mpdraw.draw_landmarks(image, titiktangan, mediapipehand.HAND_CONNECTIONS)
            # landmark penting
            thumb = titiktangan.landmark[4]   # ujung jempol
            wrist = titiktangan.landmark[0]   # pergelangan
            # jenis tangan (Left / Right)
            hand_label = hand.classification[0].label
            # LOGIKA DEPAN / BELAKANG
            status = ""
            if hand_label == "Right":
                if thumb.x < wrist.x:
                    status = "DEPAN"
                else:
                    status = "BELAKANG"
            elif hand_label == "Left":
                if thumb.x > wrist.x:
                    status = "DEPAN"
                else:
                    status = "BELAKANG"
            # TAMPILKAN HASIL
            cv2.putText(
                image,
                f"{hand_label.upper()} - {status}",(100, 50),cv2.FONT_HERSHEY_PLAIN,2.5,(0, 255, 0),3 )
    cv2.imshow("webcam", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()