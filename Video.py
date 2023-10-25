import cv2
import numpy as np

# Create a VideoCapture object to capture video from the default camera (usually 0)
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width to 1920 for full screen
cap.set(4, 1080)  # Set height to 1080 for full screen

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video capture fails

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 30, 200)  # Adjust the thresholds here

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store circle information
    circles_info = []

    # Iterate through detected contours
    for contour in contours:
        # Fit an ellipse to the contour (assuming it's a circle)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)

            # Extract ellipse information: center, axes lengths (major and minor), and angle
            center, axes, angle = ellipse

            # Calculate the diameter of the circle (average of major and minor axes)
            diameter = int((axes[0] + axes[1]) / 2.0)

            # Filter out non-circular shapes based on aspect ratio
            min_axis_length = min(axes)
            if min_axis_length > 0:
                aspect_ratio = max(axes) / min_axis_length
                if aspect_ratio < 1.2:
                    # Store circle information as a tuple (center, diameter)
                    circles_info.append((center, diameter))

    # Calculate the average diameter for each circle
    average_diameters = {}

    # Iterate through detected circles and calculate the average diameter
    for _, diameter in circles_info:
        # Find circles with similar diameters
        similar_diameters = [d for _, d in circles_info if abs(d - diameter) < 10]

        # Calculate the average diameter
        average_diameter = sum(similar_diameters) / len(similar_diameters)

        # Store the average diameter with the diameter value as the key
        average_diameters[average_diameter] = similar_diameters

    # Draw the circles and their average diameters on the frame
    for center, diameter in circles_info:
        # Convert the center coordinates to integers
        center = (int(center[0]), int(center[1]))

        # Draw the circle
        cv2.circle(frame, center, int(diameter / 2), (0, 255, 0), 3)

        # Calculate the topmost point on the circle
        topmost_point = center[1] - int(diameter / 2)

        # Find the corresponding average diameter
        for avg_diameter, similar_diameters in average_diameters.items():
            if diameter in similar_diameters:
                # Draw the average diameter as text next to the circle
                cv2.putText(frame, f"Diameter: {avg_diameter:.2f}", (center[0], topmost_point),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                break

    # Display the frame with circles and average diameters
    cv2.namedWindow("Circle Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Circle Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Circle Detection", frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
