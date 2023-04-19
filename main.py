import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math


DRAW_ALL = True
MANO = "Left"


def punto_medio_2P(p1,p2):
    (x1, y1) = p1
    (x2, y2) = p2


    pm = (int((x1+x2) / 2),int((y1+y2) / 2))
    return pm


def findAngle(p1, p2, p3):
    # da 3 punti 2d ricavo l angolo interno
    # Get the landmarks
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3

    # Calculate the Angle
    deg = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                       math.atan2(y1 - y2, x1 - x2))

    angle = np.abs(deg)
    if angle > 180:
        angle = 360 - angle

    # print(angle)

    return angle

def distance2D_tuple(P1,P2):
    (X1, Y1) = P1
    (X2, Y2) = P2
    D = math.sqrt(pow(X1-X2,2) + pow(Y1-Y2, 2))
    return D


def distance2D(P1,P2):
    X1 = P1[0]
    X2 = P2[0]
    Y1 = P1[1]
    Y2 = P2[1]
    D = math.sqrt(pow(X1-X2,2) + pow(Y1-Y2, 2))
    return D


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


def findAngle(p1, p2, p3):
    """
    findAngle : find angle from 3 points

    :param pi: the 2d points of the angle


    :return angle: the clculated angle
    """

    # ci vuole una routine che controlli geometricamente e
    # a livello di tipo che i valori di kp siano reali
    # Get the landmarks
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3

    # Calculate the Angle
    deg = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                       math.atan2(y1 - y2, x1 - x2))

    angle = np.abs(deg)
    if angle > 180:
        angle = 360 - angle

    # print(angle)

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
"""
std_x = 0
std_y = 0
media_x = 0
media_y = 0

"""

#plt.axis([0, 10000, 0, 1])

# For webcam input:
#telecamera
path_2_video = "/home/mmt-ben/HAND_WASH/video/IMG_2902.MOV"
cap = cv2.VideoCapture(path_2_video)

baricentro_x = []
baricentro_y = []
distanze = []
angoli = []
faccia_mano = []
distanze_mignoli_indici_medi  = []
distanze_basricentri = []
angoli_nc = []


with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    i = 0

    #ciclo streaming
    while cap.isOpened():


        #lettura foto:
        success, image = cap.read()

        if i >= 20:

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break


        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
            image.flags.writeable = False

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #rete neurale google
            results = hands.process(image)



            #plt.scatter(i, y)
            #plt.pause(0.00005)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #print(image.shape)
            h,w,c = image.shape



            if results.multi_hand_landmarks:
                if DRAW_ALL:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())




                #punto nocca dx

                if True:
                #if (len(results.multi_hand_landmarks)) > 1:
                    #questo ciclo viene iterato su ogni mano
                    punti_nocca_indice = []
                    punti_x_angolo = []
                    x_mano = []
                    indici_e_mignoli = []
                    bar1_punti_1 = [[],[]]
                    bar1_punti_2 = [[],[]]
                    for j in range(len(results.multi_hand_landmarks)):


                                nocche_ind_medio = []
                                punti_angolo_ind_medio = []





                                # label = left or right
                                label = results.multi_handedness[j].classification[0].label
                                #print("mano : ", label)


                                data = []
                                datay = []
                                dito_due_punti = []

                                for index, landmark in enumerate(results.multi_hand_landmarks[j].landmark):


                                    #
                                    # if index == 8 or index == 20:
                                    #     indici_e_mignoli.append(((int(landmark.x * w), int(landmark.y * h))))
                                    #     cv2.circle(image, ((int(landmark.x * w), int(landmark.y * h))), 10,
                                    #                (0, 255, 255), 3)
                                    #
                                    #
                                    #
                                    # if label == MANO:
                                    #     if index == 8 or index == 12 or index == 16 or index == 20:
                                    #         bar1_punti_1[0].append(int(landmark.x * w))
                                    #         bar1_punti_1[1].append(int(landmark.y * h))
                                    #         cv2.circle(image, ((int(landmark.x * w), int(landmark.y * h))), 10,
                                    #                    (0, 255, 255), 3)
                                    # if label != MANO:
                                    #     if index == 0 or index == 1 or index == 5 or index == 9 or index == 13 or index == 17:
                                    #         bar1_punti_2[0].append(int(landmark.x * w))
                                    #         bar1_punti_2[1].append(int(landmark.y * h))
                                    #         cv2.circle(image, ((int(landmark.x * w), int(landmark.y * h))), 10,
                                    #                    (0, 0, 255), 3)
                                    #
                                    #







                                    # if label == MANO:
                                    #     if index == 4 or index == 20:
                                    #         x_mano.append(int(landmark.x * w))
                                    #         cv2.circle(image, ((int(landmark.x * w), int(landmark.y * h))), 10,
                                    #                    (0, 255, 255), 3)
                                    #


                                    if index == 8 or index == 12:
                                        if label == MANO:
                                            cv2.circle(image, ((int(landmark.x * w),int(landmark.y * h))), 10, (0, 255, 255), 3)
                                            punti_angolo_ind_medio.append((int(landmark.x * w),int(landmark.y * h)))



                                    # if index == 0 or index == 8:
                                    #
                                    #     estremità_dito = [int(landmark.x * w),int(landmark.y * h)]
                                    #     dito_due_punti.append(estremità_dito)
                                        #cv2.circle(image, tuple(estremità_dito), 10, (0, 255, 0), 3)
                                    #
                                    if index == 5 or index == 9:
                                        #nocche medio e indice
                                        if label == MANO:
                                            nocche_ind_medio.append((int(landmark.x * w),int(landmark.y * h)))
                                            #cv2.circle(image, ((int(landmark.x * w),int(landmark.y * h))), 10, (0, 255, 255), 3)



                                        # cv2.circle(image, tuple(estremità_dito), 10, (0, 255, 0), 3)
                                    #
                                    # #5,6,8 x ANGOLO
                                    # if index == 5 or index == 6 or index == 8:
                                    #     if label == MANO:
                                    #         punti_x_angolo.append((int(landmark.x * w),int(landmark.y * h)))
                                    #         cv2.circle(image, ((int(landmark.x * w), int(landmark.y * h))), 10,
                                    #                    (0, 255, 0), 3)
                                    #




                                    #confronto distanza due mani
                                    # if index == 5:
                                    #
                                    #
                                    #     punto = (int(landmark.x * w),int(landmark.y * h))
                                    #
                                    #     #cv2.circle(image, punto, 10,(0, 255, 0), 3)
                                    #     punti_nocca_indice.append(punto)
                                    #     cv2.circle(image, ((int(landmark.x * w), int(landmark.y * h))), 10,
                                    #                (0, 255, 0), 3)



                                        #calcoliamo  distanza nocche degli indici
                                        #punto joints = 5
                                        #calcoliamo distanza nocche pollici
                                        #mano chiusa ->
                                        """
                                        med_x = sum(data) / len(data)
                                        med_y = sum(datay) / len(datay)
                                        baricentro_x.append(med_x)
                                        baricentro_y.append(med_y)
                                        """


                    #
                    # if len(indici_e_mignoli) > 3:
                    #     dist_1 = distance2D_tuple(indici_e_mignoli[0], indici_e_mignoli[3])
                    #     dist_2 = distance2D_tuple(indici_e_mignoli[1], indici_e_mignoli[2])
                    #     dist = np.mean([dist_1,dist_2])
                    #     distanze_mignoli_indici_medi.append(dist)
                    # distanze_mignoli_indici_medi.append(0)
                    #
                    #
                    #
                    #
                    # if len(bar1_punti_2[0]) != 0 and len(bar1_punti_1[0]) != 0:
                    #     med_x = int(sum(bar1_punti_2[0]) / len(bar1_punti_2[0]))
                    #     med_y = int(sum(bar1_punti_2[1]) / len(bar1_punti_2[1]))
                    #     med_x_n = int(sum(bar1_punti_1[0]) / len(bar1_punti_1[0]))
                    #     med_y_n = int(sum(bar1_punti_1[1]) / len(bar1_punti_1[1]))
                    #     print(bar1_punti_2, bar1_punti_1)
                    #     dist = distance2D_tuple((med_x,med_y),(med_x_n,med_y_n))
                    #     cv2.line(image, (med_x,med_y),(med_x_n,med_y_n), (255, 0, 0), 3)
                    # else:
                    #     dist = 0
                    # distanze_basricentri.append(dist)



                    #fine ciclo ogni joins mano
                    if len(nocche_ind_medio) > 1:
                        Pmedio = punto_medio_2P(nocche_ind_medio[0],nocche_ind_medio[1])

                        #cv2.circle(image, Pmedio, 8, (0, 0, 255), 3)

                        punti_angolo_ind_medio.append(Pmedio)

                    #
                    # if len(punti_x_angolo) > 2:
                    #     #print(punti_angolo_ind_medio      )
                    #     angolo = findAngle(punti_x_angolo[0], punti_x_angolo[1], punti_x_angolo[2])
                    #     angoli.append(angolo)
                    # else:
                    #     angoli.append(0)

                    if len(punti_angolo_ind_medio) > 2:
                        #print(punti_angolo_ind_medio      )
                        angolo_nc = findAngle(punti_angolo_ind_medio[0], punti_angolo_ind_medio[2], punti_angolo_ind_medio[1])
                        angoli_nc.append(angolo_nc)
                    else:
                        angoli_nc.append(0)




                    # if label == MANO:
                    #     print(label)
                    #     lunghezza_dito = distance2D(dito_due_punti[0], dito_due_punti[1])
                    #     #cv2.line(image, (dito_due_punti[0][0],dito_due_punti[0][1]), (dito_due_punti[1][0],dito_due_punti[1][1]), (255, 0, 0), 3)
                    #
                    #     if len(punti_x_angolo) > 2:
                    #         angle = findAngle(punti_x_angolo[0],punti_x_angolo[1],punti_x_angolo[2] )
                    #     else:
                    #         angle = 0



                    #
                    # if len(punti_nocca_indice) > 1:
                    #     distanza = distance2D_tuple(punti_nocca_indice[0], punti_nocca_indice[1])
                    #     distanze.append(distanza)
                    #     cv2.line(image, (punti_nocca_indice[0][0],punti_nocca_indice[0][1]), (punti_nocca_indice[1][0],punti_nocca_indice[1][1]), (255, 0, 0), 3)
                    # else:
                    #     distanze.append(0)

                    #

                    # if len(x_mano) == 0:
                    #     x_mano = [0,0]
                    # delta = x_mano[0] - x_mano[1]
                    # if x_mano[0] == 0:
                    #     faccia_mano.append(-5)
                    # else:
                    #     if delta > 100:
                    #         faccia_mano.append(1)
                    #     elif delta < -100:
                    #         faccia_mano.append(-1)
                    #     else:
                    #         faccia_mano.append(0)



        i = i + 1
        #print(i)
        # Flip the image horizontally for a selfie-view display.
        image_res = resize_image(image, 50)
        cv2.imshow('MediaPipe Hands', cv2.flip(image_res, 1))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed

cap.release()


# std_x = np.std(baricentro_x)
# std_y = np.std(baricentro_y)
# media_x = np.mean(baricentro_x)
# media_y = np.mean(baricentro_y)


titlee = "Angolo indice medio, estensione mano"
plt.title(str(titlee))
#plt.scatter(baricentro_x,baricentro_y)
#plt.scatter(np.linspace(1,len(distanze), len(distanze)),distanze)

quantita = angoli_nc
plt.scatter(np.linspace(1,len(quantita), len(quantita)),quantita)
plt.xlabel('frame [s]')
plt.ylabel('angolo [°]')
plt.show()
