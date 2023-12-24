from features import (eye_feature, mouth_feature,
                      area_eye_feature, area_mouth_feature,
                      pupil_feature, mp_face_mesh, mp_drawing, SHOW_MESH, connections_drawing_spec)
import xgboost
import numpy as np
import cv2
import pandas as pd
import joblib
from limited_array import LimitedSizeArray

loaded_model = xgboost.Booster()
loaded_model.load_model('models/xgb/xgb.xgb')
loaded_scaler = joblib.load('models/scaler/standard_scaler.joblib')

frame_count = 0

cap = cv2.VideoCapture(0)

limited_array_size = 16
check_awake = LimitedSizeArray(limited_array_size)

while True:
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        success, image = cap.read()
        if not success:
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            landmarks_positions = []
            # assume that only face is present in the image
            for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                landmarks_positions.append(
                    [data_point.x, data_point.y, data_point.z])  # saving normalized landmark positions

            landmarks_positions = np.array(landmarks_positions)
            landmarks_positions[:, 0] *= image.shape[1]
            landmarks_positions[:, 1] *= image.shape[0]

            eye = eye_feature(landmarks_positions)
            mouth = mouth_feature(landmarks_positions)
            area_eye = area_eye_feature(landmarks_positions)
            area_mouth = area_mouth_feature(landmarks_positions)
            pupil = pupil_feature(landmarks_positions)

            features = loaded_scaler.transform(pd.DataFrame({
                'eye': [eye],
                'mouth': [mouth],
                'area_eye': [area_eye],
                'area_mouth': [area_mouth],
                'pupil': [pupil]
            }))

            prediction = loaded_model.predict(xgboost.DMatrix(features))

            check_awake.push(0 if prediction[0] < 0.5 else 1)
            label = 'Tired'
            if check_awake.count_zeros() >= limited_array_size / 2:
                label = 'Awake'

            # Вывод результата на кадр
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=SHOW_MESH,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connections_drawing_spec)

            cv2.imshow('Video', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
