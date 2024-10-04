import cv2
import streamlit as st
import numpy as np
import os

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(scale_factor, min_neighbors, rect_color):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Create a placeholder in Streamlit for the video frame
    stframe = st.empty()
    
    # Continuously capture frames until the user stops detection
    while st.session_state.is_detecting:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video")
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

        # Convert the frame to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the image in the Streamlit app
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Add a small delay to control the frame rate
        # cv2.waitKey(1) # Removed this line to avoid GUI function error

    # Release the capture when done
    cap.release()
    # Removed cv2.destroyAllWindows() since it's not needed in Streamlit

def main():
    st.title("Real-Time Face Detection using Viola-Jones Algorithm")

    # Initialize session state if not already initialized
    if 'is_detecting' not in st.session_state:
        st.session_state.is_detecting = False

    st.write(""" 
        ## Overview 
        This app uses the Viola-Jones algorithm for real-time face detection. 
        Adjust the parameters to customize the detection and see the results in real time. 
        📸👤
    """)

    st.write(""" 
        ## Instructions 
        1. Adjust the parameters using the sliders below. 
        2. Choose the color of the rectangles using the color picker. 
        3. Press the "Detect Faces" button to start detecting faces from your webcam. 
        4. Click "Stop Detection" to stop the face detection. 
    """)

    scale_factor = st.slider("Scale Factor", 1.01, 2.0, 1.3, 0.01)
    min_neighbors = st.slider("Min Neighbors", 1, 10, 5, 1)
    rect_color = st.color_picker("Rectangle Color", "#00FF00")

    # Convert hex color to BGR format for OpenCV
    rect_color = tuple(int(rect_color[i:i+2], 16) for i in (5, 3, 1))  # Convert hex to BGR (reverse order)

    # Start detection button
    if st.button("Detect Faces"):
        st.session_state.is_detecting = True
        detect_faces(scale_factor, min_neighbors, rect_color)

    # Display stop button only if detecting
    if st.session_state.is_detecting:
        # Show the stop button while detection is ongoing
        if st.button("Stop Detection"):
            st.session_state.is_detecting = False

    # Check if the app is running in a cloud environment
    if 'CLOUD_ENV' in os.environ:
        st.warning("Webcam access is not available in the cloud environment. Please run this app locally for video capture.")

if __name__ == "__main__":
    main()
