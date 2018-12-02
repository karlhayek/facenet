import argparse, sys, copy, operator, time, cv2
import face


def updatePrevsFaces(faces, prevs_faces, frame_interval , non_cont_max):
    new_prevs_faces = faces
    if(len(prevs_faces) > 0):
        for prevs_face in prevs_faces:
            overlapped_with_any_face = False
            for face in faces:
                if isOverlap(face.bounding_box, prevs_face.bounding_box):
                    overlapped_with_any_face = True
                    face.non_cont_frames = 0

            if not overlapped_with_any_face and prevs_face.non_cont_frames < non_cont_max:
                prevs_face.non_cont_frames = prevs_face.non_cont_frames + frame_interval  
                new_prevs_faces.append(prevs_face)

    return new_prevs_faces


def isOverlap(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an xis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou > 0.2

def getDict(face, prevs_faces_list, prevs_direct_faces_list, frame_interval):
    if(len(prevs_faces_list) > 0):
        for prevs_face in prevs_faces_list:
            if isOverlap(face.bounding_box, prevs_face.bounding_box):
                #add to dict
                face.dic_name_count = prevs_face.dic_name_count

                if face.name not in face.dic_name_count.keys():
                    face.dic_name_count[face.name] = 1
                else:
                    face.dic_name_count[face.name] += 1

                #Add continous frames
                face.cont_frames = prevs_face.cont_frames + frame_interval

                if(face.name != prevs_face.name):
                    face.name = max(face.dic_name_count.items(), key=operator.itemgetter(1))[0]
    return face

def getFaces(faces, prevs_faces_list, prevs_direct_faces_list, frame_interval):
    for face_ in faces:
        face_ = getDict(face_, prevs_faces_list, prevs_direct_faces_list, frame_interval)

        print('\n     cont : ' + str(face_.cont_frames))
        print('     non_cont : ' + str(face_.non_cont_frames))
        print('     Dictionary : ')
        for k,v in face_.dic_name_count.items():
            print('         ' +str(k) + ' -> ' + str(v))
        print('----------------------------------------')    

    return faces


def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)


def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    frame_rate = 0
    frame_count = 0
    video_capture = cv2.VideoCapture(0)
    width, height = 640, 480
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    face_recognition = face.Recognition()

    prevs_faces_list, prevs_direct_faces_list = [], []

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Fix recognition for consecutive frames
            faces = getFaces(faces, prevs_faces_list, prevs_direct_faces_list, frame_interval)
            prevs_direct_faces_list = faces
            prevs_faces_list = updatePrevsFaces(copy.deepcopy(faces), prevs_faces_list, frame_interval, 5)

        add_overlays(frame, faces)

        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
