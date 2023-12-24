import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture('./tired/2022-12-25 23:44:59.873050.mp4')

right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]  # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]  # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]  # mouth landmark coordinates

SHOW_MESH = frozenset([
    (33, 133), (160, 144), (159, 145), (158, 153),
    (263, 362), (387, 373), (386, 374), (385, 380),
    (61, 291), (39, 181), (0, 17), (269, 405)
])

connections_drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=3,
    color=(255, 255, 255)
)


def distance(p1, p2):
    return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5


def eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)


def eye_feature(landmarks):
    return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye)) / 2


def mouth_feature(landmarks):
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)


def perimeter(landmarks, eye):
    return distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
           distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
           distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
           distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
           distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
           distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
           distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
           distance(landmarks[eye[1][1]], landmarks[eye[0][0]])


def perimeter_feature(landmarks):
    return (perimeter(landmarks, left_eye) + perimeter(landmarks, right_eye)) / 2


def area_eye(landmarks, eye):
    return math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)


def area_mouth_feature(landmarks):
    return math.pi * ((distance(landmarks[mouth[1][0]], landmarks[mouth[3][1]]) * 0.5) ** 2)


def area_eye_feature(landmarks):
    return (area_eye(landmarks, left_eye) + area_eye(landmarks, right_eye)) / 2


def pupil_circularity(landmarks, eye):
    return (4 * math.pi * area_eye(landmarks, eye)) / (perimeter(landmarks, eye) ** 2)


def pupil_feature(landmarks):
    return (pupil_circularity(landmarks, left_eye) +
            pupil_circularity(landmarks, right_eye)) / 2
