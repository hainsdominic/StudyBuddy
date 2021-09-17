import cv2
import time
import pickle

eyeglasses_cascade = cv2.CascadeClassifier(
    "cascades/haarcascade_eye_tree_eyeglasses.xml"
)
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")


def detectEyes():
    timestamp = None
    cap = cv2.VideoCapture(0)
    distractions = {}

    while True:
        now = int(time.time())

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

        # Add the attention status to the distractions dictionnay
        if len(eyeglasses) + len(eye) == 0:
            # Get the current timestamp in seconds
            current = int(time.time())

            if timestamp == None:
                timestamp = current

            if current > timestamp:
                distractions[current] = 1
                timestamp = None
        else:
            if now not in distractions:
                distractions[now] = 0
            timestamp = None

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

    return distractions


def plot_distractions(distractions: dict):
    import matplotlib.pyplot as plt  # Avoids seg. error

    # Data for plotting
    x = list(key - next(iter(distractions)) for key in distractions.keys())
    y = list(distractions.values())

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(
        xlabel="Timestamp (s)",
        ylabel="Distractions",
        title="Distractions versus time",
    )

    plt.yticks([0, 1])

    fig.savefig("saves/" + str(next(iter(distractions))) + ".png")

    plt.show()


def save_obj(obj):
    with open("saves/" + str(next(iter(obj))) + "_distractions" + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    distractions_observed = detectEyes()
    plot_distractions(distractions_observed)
    save_obj(distractions_observed)
