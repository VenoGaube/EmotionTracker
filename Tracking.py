import cv2
import dlib


def main():
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()

        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)

        for face in faces:
            # Create landmark object
            landmarks = predictor(image=gray, box=face)

            # Loop through all the points
            left_x, left_y, right_x, right_y, left_lip_x, left_lip_y, right_lip_x, right_lip_y = 0, 0, 0, 0, 0, 0, 0, 0

            for n in range(48, 60):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                if n == 48:
                    left_x = x
                    left_y = y

                if n == 54:
                    right_x = x
                    right_y = y

                if n == 49:
                    left_lip_x = x
                    left_lip_y = y

                if n == 53:
                    right_lip_x = x
                    right_lip_y = y

                # Draw a circle
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-2)
            # print("\r Left corner: " + str(left_x) + ", " + str(left_y) + " Right corner: " + str(right_x) + ", " + str(right_y) + " Left diffy: " + str(left_y-left_lip_y) + " Right diffy: " + str(right_y-right_lip_y), end=" ")
            if left_y - left_lip_y <= 3 and right_y - right_lip_y <= 3:
                print(":)")
            if left_y - left_lip_y >= 11 and right_y - right_lip_y >= 11:
                print(":(")
        cv2.imshow("Live Feed", frame)

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
