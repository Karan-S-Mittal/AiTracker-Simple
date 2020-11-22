import cv2  # Import OpenCV as cv2

video = cv2.VideoCapture(0)  # Save the webcam feed or video as a variable

# files from librry of opencv
car_tracker_file = 'cars.xml'
car_tracker = cv2.CascadeClassifier(car_tracker_file)

pedestrian_tracker_file = 'pedestrians.xml'
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:

    (success, frame) = video.read()

    if success:
        # Convert the image to a grayscale image
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Store the grayscale frames
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Find the coordinates of the objects
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('AI Driven Car Tracker', frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

video.release()
