import cv2
import streamlit as st
import numpy as np

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Real-Time Face Detection using Viola-Jones Algorithm")

    # Initialize session state if not already initialized
    if 'is_detecting' not in st.session_state:
        st.session_state.is_detecting = False

    st.write("""
        ## Overview
        This application demonstrates real-time face detection using the Viola-Jones algorithm.
        It utilizes OpenCV's pre-trained Haar Cascade classifier to detect faces from your webcam feed.
        You can adjust the parameters to fine-tune the detection process.

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

    # Display stop button only if detecting
    if st.session_state.is_detecting:
        st.button("Stop Detection", on_click=stop_detection)

    # Run face detection if is_detecting is True
    if st.session_state.is_detecting:
        detect_faces(scale_factor, min_neighbors, rect_color)
    
    st.write("""
        ## About the Developer
        **Name:** Ghazi Lassoued
        **LinkedIn:** [Ghazi Lassoued](https://www.linkedin.com/in/ghazi-lassoued-983419239/)
        **Email:** lassouedghazi21@gmail.com
    """)

def stop_detection():
    st.session_state.is_detecting = False

if __name__ == "__main__":
    main()

