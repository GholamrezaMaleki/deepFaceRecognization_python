import numpy as np
import cv2
import matplotlib.pyplot as plt
import args


# this function will visualize face in webcam
def visualize(input, faces, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0),
                          thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)


# reading deep learning model to detect face
detector = cv2.FaceDetectorYN.create(
    "model/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# face to compare with face in webcam
image1 = cv2.imread("images/ali_sadeghi.webp")
img1 = image1.copy()
img1Width = int(img1.shape[1])
img1Height = int(img1.shape[0])
img1 = cv2.resize(img1, (img1Width, img1Height))

detector.setInputSize((img1Width, img1Height))
faces1 = detector.detect(img1)

assert faces1[1] is not None, 'Cannot find a face in {}'.format(args.image1)
# Draw results on the input image
visualize(img1, faces1)

# reading deep learning model to recognize face
recognizer = cv2.FaceRecognizerSF.create(
    "model/face_recognition_sface_2021dec.onnx", "")

face1_align = recognizer.alignCrop(image1, faces1[1][0])
plt.imshow(face1_align[..., ::-1])
plt.show()
face1_feature = recognizer.feature(face1_align)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frm1 = frame.copy()
    frm1Width = int(frm1.shape[1])
    frm1Height = int(frm1.shape[0])
    frm1 = cv2.resize(frm1, (frm1Width, frm1Height))
    detector.setInputSize((frm1Width, frm1Height))
    faces2 = detector.detect(frm1)
    if faces2[1] is not None:
        visualize(frm1, faces2)

    face2_align = recognizer.alignCrop(frame, faces2[1][0])
    face2_feature = recognizer.feature(face2_align)

    l2_similarity_threshold = 1.128

    l2_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_NORM_L2)

    msg = 'different identities'
    if l2_score <= l2_similarity_threshold:
        msg = 'the same identity'
    print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg,
                                                                                                                    l2_score,
                                                                                                                    l2_similarity_threshold))

    cv2.imshow("frame", frm1)
    if ((cv2.waitKey(1) & 0xFF) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
