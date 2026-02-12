import cv2
import mediapipe as mp

# Inisialisasi kamera
capture = cv2.VideoCapture(0)

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
tangan = mp_hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = capture.read()
    if not success:
        break

    # Konversi ke RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for titiktangan in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                titiktangan,
                mp_hands.HAND_CONNECTIONS
            )

            # Ambil koordinat tiap titik
            for id, titikt in enumerate(titiktangan.landmark):
                print(id, titikt.x, titikt.y)

    # Tampilkan kamera
    cv2.imshow("Hand Detection", img)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()