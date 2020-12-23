import os
import shutil
import sys
import time

sys.path.append("/home/ash/face_landmark_factory/")

import cv2
import numpy as np

from testing.mark_detector import MarkDetector


def webcam_main():
    print("Camera sensor warming up...")

    mark_detector = MarkDetector(current_model, CNN_INPUT_SIZE)
    if current_model.split(".")[-1] == "pb":
        run_model = 0
    elif current_model.split(".")[-1] == "hdf5" or current_model.split(".")[-1] == "h5":
        run_model = 1
    else:
        print("input model format error !")
        return

    # Set up directory where images will be saved
    if os.path.exists('saved_images'):
        shutil.rmtree('saved_images')
    os.mkdir('saved_images')

    full_face = cv2.imread('aiface1.jpg')
    height = 750
    width = 750

    full_face = cv2.resize(full_face, (height, width))
    height_coordinates = np.arange(height)
    width_coordinates = np.arange(width)
    hw_coordinates, wh_coordinates = np.meshgrid(height_coordinates, width_coordinates)

    hw_coordinates = np.reshape(hw_coordinates, -1)
    wh_coordinates = np.reshape(wh_coordinates, -1)


    shuffled_coordinates = np.random.permutation(len(hw_coordinates))

    face_mask = np.zeros([height, width, 3], dtype=np.uint8)

    counter = 0

    for i in shuffled_coordinates :
        height_coord = hw_coordinates[i]
        width_coord = wh_coordinates[i]
        face_mask[height_coord, width_coord, :] = 1

        if counter % 30000 == 0 or i == len(shuffled_coordinates)-1:
            frame = full_face.copy()
            frame *= face_mask

            faceboxes = mark_detector.extract_cnn_facebox(frame)

            if faceboxes is not None:
                for facebox in faceboxes:
                    # Detect landmarks from image of 64X64 with grayscale.
                    face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]
                    cv2.rectangle(
                        frame,
                        (facebox[0], facebox[1]), (facebox[2], facebox[3]),
                        (0, 255, 0),
                        2)
                    face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_img0 = face_img.reshape(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 1)

                    if run_model == 1:
                        marks = mark_detector.detect_marks_keras(face_img0)
                    else:
                        marks = mark_detector.detect_marks_tensor(face_img0, 'input_2:0', 'output/BiasAdd:0')
                    # marks *= 255
                    marks *= facebox[2] - facebox[0]
                    marks[:, 0] += facebox[0]
                    marks[:, 1] += facebox[1]
                    # Draw Predicted Landmarks
                    mark_detector.draw_marks(frame, marks, color=(255, 255, 255), thick=2)


            # Save the frame, detections, and landmarks as separate images
            cv2.imwrite(
                'saved_images/modified_frame_{}.png'.format(counter),
                frame)

            print('Just processed frame {}'.format(counter))
        counter += 1


    # do a bit of cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
#    current_model = "../model/facial_landmark_cnn.h5"
    current_model = "/home/ash/face_landmark_factory/model/facial_landmark_cnn.pb"
#    VIDEO_PATH = 0
    CNN_INPUT_SIZE = 64
    webcam_main()
