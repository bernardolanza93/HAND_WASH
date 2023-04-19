import os
import sys

import cv2
print(cv2.__version__)
import csv
import mediapipe as mp



def resize_image(img, percentage):
    # Display the frame
    # saved in the file
    scale_percent = 70  # percent of original size
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized



def header_generator():
    """
    header_generator : generate the headers of the csv file
    :param nothing:
    :return label_data: the headers of the csv file
    """

    label_data = []
    for i in range(21):
        label_data.append('xl' + str(i))
        label_data.append('yl' + str(i))
        label_data.append('zl' + str(i))

    for i in range(21):
        label_data.append('xr' + str(i))
        label_data.append('yr' + str(i))
        label_data.append('zr' + str(i))
    #print(label_data)
    return label_data


def single_class_video_mediapipe_raw():
    print("start")


    #mediapipe settings
    empty = []
    MOD_COMPLX = 1
    MINTRKCONF = 0.5
    MINDETCONF = 0.1
    WRITE_LABEL_VIDEO = 1
    SIMh = False  # static_image_mode
    DRAW = True
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScaleq
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2



    for zero in range(63):
        empty.append(float('Nan'))
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Set the directory containing the videos
    video_directory = "single moves/"

    # Set the directory for saving the CSV files
    output_directory = "mediapipe_raw_data/"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Loop through each video file in the directory
    for filename in os.listdir(video_directory):
        if filename.endswith(".MOV"):

            # Extract the video information from the filename
            print("file:", filename)
            filename_arr = list(filename)
            test_num = ""
            sub_num = ""
            class_num = ""

            for i in range(len(filename_arr)):
                if filename_arr[i].isalpha():
                    if filename_arr[i] == "t":
                        test_num = "".join(filename_arr[i + 1:i + 4])
                    elif filename_arr[i] == "s":
                        sub_num = "".join(filename_arr[i + 1:i + 4])
                    elif filename_arr[i] == "c":
                        class_num = "".join(filename_arr[i + 1:i + 4])

            # Open the video file
            video_file = os.path.join(video_directory, filename)
            video = cv2.VideoCapture(video_file)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/4)
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/4)
            size = (width,height)
            #fcc = video.get(cv2.CV_CAP_PROP_FOURCC)
            print(width,height)
            # Get video information
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("total frame:", frame_count)
            fps = float(video.get(cv2.CAP_PROP_FPS))
            print(fps)
            duration = frame_count / fps
            print("duration:", duration)
            filename_no_ext = filename.split(".")[0]
            print(filename_no_ext)

            # Create a CSV file for the video
            output_filename = f"t{test_num}_s{sub_num}_c{class_num}_{int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}_{int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}.csv"
            output_path = os.path.join(output_directory, output_filename)
            if WRITE_LABEL_VIDEO:

                fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
                out = cv2.VideoWriter("single moves with annotations" + "/" + "annotated_" +  filename, cv2.VideoWriter_fourcc(*'XVID'), fps,size)  # setta la giusta risoluzionw e fps

                #out = cv2.VideoWriter("single moves with annotations" + "/" + "annotated_" +  , fourcc, 30.0,(int(video.get(3)),int(video.get(4))),True)  # setta la giusta risoluzionw e fps
            with open(output_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                kp_headers = header_generator()
                writer.writerow(["f", "time","t", "s", "c"] + kp_headers + ["hnl", "hnr"])

                with mp_hands.Hands(
                        static_image_mode=SIMh,
                        model_complexity=MOD_COMPLX,
                        min_detection_confidence=MINDETCONF,
                        min_tracking_confidence=MINTRKCONF) as hands:
                    # Loop through each frame in the video

                    while True:
                        # Read the next frame
                        ret, frame = video.read()
                        hdnR = 0
                        hdnL = 0

                        # Stop if end of video is reached
                        if not ret:
                            break


                        # Extract frame number and timestamp
                        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                        timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000



                        #mediapipe
                        frame.flags.writeable = False
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        results = hands.process(frame)
                        frame.flags.writeable = True
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())

                            # dimension of the image and copy to annotate
                        image_height, image_width, _ = frame.shape



                        if not results.multi_hand_landmarks or not results.multi_handedness:
                            leftKP = empty
                            rightKP = empty
                            all_KP = leftKP + rightKP
                        else:
                            n_mani = len(results.multi_hand_landmarks)
                            leftKP = []
                            rightKP = []

                            for j in range(len(results.multi_hand_landmarks)):

                                # label = left or right
                                label = results.multi_handedness[j].classification[0].label
                                handedness = results.multi_handedness[j].classification[0].score

                                keypoints = []
                                # extract all keypoint of the hand

                                for index, landmark in enumerate(results.multi_hand_landmarks[j].landmark):

                                    keypoints.append(round(landmark.x,5))
                                    keypoints.append(round(landmark.y,5))
                                    keypoints.append(round(landmark.z,5))

                                # divide by left or right hand
                                if label == 'Left':
                                    leftKP = keypoints
                                    hdnL = handedness
                                if label == 'Right':
                                    rightKP = keypoints
                                    hdnR = handedness

                            # feed empty list if hand is not present
                            if leftKP == []:
                                leftKP = empty

                            if rightKP == []:
                                rightKP = empty

                            all_KP = leftKP + rightKP

                            # insert the ML label into the data to be written

                            # write a row of data


                        # Write the frame information to the CSV file
                        writer.writerow([frame_num, round(timestamp,5),str(int(test_num)),str(int(sub_num)), class_num]+all_KP+[hdnL,hdnR])
                        if DRAW:
                            image = cv2.putText(frame, "frame: "+ str(frame_num) + " time:" + str(round(timestamp,2)), org, font,
                                                fontScale, color, thickness, cv2.LINE_AA)
                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    mp_drawing.draw_landmarks(
                                        frame,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                            frame = resize_image(frame, 25)
                            if WRITE_LABEL_VIDEO:
                                out.write(frame)



                            cv2.imshow('frame', frame)
                            key = cv2.waitKey(1)
                            if key == ord('q'):
                                break
                            if key == ord('p'):
                                cv2.waitKey(-1)  # wait until any key is pressed

            # Release the video file
            video.release()
            if WRITE_LABEL_VIDEO:
                out.release()

    print("Done.")



single_class_video_mediapipe_raw()