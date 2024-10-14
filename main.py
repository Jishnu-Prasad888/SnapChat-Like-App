import cv2
import cvzone

def capture_pic(frame): 
    cv2.imwrite('captured_image.jpg', frame)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
overlay = cv2.imread('beard.png', cv2.IMREAD_UNCHANGED)
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
smile_detected = False

while True: 
    ret, frame = cap.read()

    if not ret: 
        print("Error: could not read video source.")
        break
    
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale)

    for (x, y, w, h) in faces: 
        roi_gray = gray_scale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        overlay_resize = cv2.resize(overlay,(int(w*1.8),int(h*1.8)))
        frame = cvzone.overlayPNG(frame,overlay_resize,[x-45,y-70])

        smile_region = roi_gray[int(h / 2):h, 0:w]
        smiles = smile_cascade.detectMultiScale(smile_region, scaleFactor=1.8, minNeighbors=20)

        if len(smiles) > 0 and not smile_detected: 
            capture_pic(frame)
            smile_detected = True
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy + int(h / 2)), (sx + sw, sy + int(h / 2) + sh), (0, 255, 0), 2)
        elif len(smiles) == 0:
            smile_detected = False  # Reset the flag when no smile is detected

    cv2.imshow("Snap Chat", frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()