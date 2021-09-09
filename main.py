import cv2
from playsound import playsound
from time import time

eyeglasses_cascade = cv2.CascadeClassifier(
    "cascades/haarcascade_eye_tree_eyeglasses.xml"
)
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")


def detectEyes():
    timestamp = None
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame by frame
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyeglasses = eyeglasses_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5
        )
        eye = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in eyeglasses:
            color = (255, 0, 0)  # BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        for (x, y, w, h) in eye:
            color = (255, 255, 0)  # BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Display resulting frame
        cv2.imshow("frame", frame)

        if len(eyeglasses) + len(eye) == 0:
            # Get the current timestamp in seconds
            current = int(time())
            if timestamp == None:
                timestamp = current

            if current > timestamp + 2:
                playsound("alarm.wav")
                timestamp = None
        else:
            timestamp = None

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectEyes()
