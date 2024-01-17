import cv2
import streamlit as st
import os

face_cascade = cv2.CascadeClassifier(r'C:\Users\gunx0\Desktop\MyPyCodes\faceR\haarcascade_frontalface_default .xml')

# directory to save the detected faces
output_directory = r'C:\Users\gunx0\Desktop\MyPyCodes\faceR\detected_faces'
os.makedirs(output_directory, exist_ok=True)

def hex_to_rgb(hex_color):
    """Converts hexadecimal color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def detect_faces(min_neighbors, scale_factor, rectangle_color):
    cap = cv2.VideoCapture(0)
    counter = 0

    while True:
        ret, frame = cap.read()

        # to Check if the frame is not empty and the webcam is ready
        if not ret:
            st.warning("Unable to read frame. Please check your webcam connection.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        for (x, y, w, h) in faces:
            rectangle_color_bgr = hex_to_rgb(rectangle_color)[::-1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color_bgr, 2)


            filename = os.path.join(output_directory, f'detected_face_{counter}.jpg')
            cv2.imwrite(filename, frame[y:y+h, x:x+w])
            counter += 1

        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # 'q' is not working with me so i make it 10 limit of specific number of faces
        if cv2.waitKey(1) & 0xFF == ord('q') or counter >= 10:
            break

    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")

    st.write("Adjust the parameters and color using the sliders and color picker below:")
    
    min_neighbors = st.slider("minNeighbors", min_value=3, max_value=10, value=5)
    scale_factor = st.slider("scaleFactor", min_value=1.1, max_value=2.0, step=0.1, value=1.3)

    rectangle_color = st.color_picker("Rectangle Color", value="#00FF00")

    if st.button("Detect Faces"):
        detect_faces(min_neighbors, scale_factor, rectangle_color)

if __name__ == "__main__":
    app()


