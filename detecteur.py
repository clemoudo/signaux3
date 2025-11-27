import cv2
import numpy as np
import time
import classes
from numpy.ma.extras import average

cv2.namedWindow("preview")

vc = cv2.VideoCapture(0)

person = []

xmid = 0
ymid = 0
show_box = False
VP = classes.vieux(xmid,ymid,[0,0,0,0])

fps = vc.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
frame_duration = 1.0 / fps

rval, frame = vc.read()
if not rval:
    exit()

target_w, target_h = 1080,720
frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

counter = 1
potential_fall = False


start_time = time.time()
frame_idx = 1

check_frame = ((0,500),(0,500))

last_bed_frame = None

while True:
    rval, frame = vc.read()
    if not rval:
        break

    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Détection de mouvement par différence de frames ---
    diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    if not VP.still:
        motion_mask_ratio = np.count_nonzero(motion_mask) / motion_mask.size
    if counter % 10 != 0:
        counter += 1
    else:
        counter =0
        ys, xs = np.where(motion_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            xmid = (x_min + x_max) // 2
            ymid = (y_min + y_max) // 2
        if motion_mask_ratio > 0.0005:
            VP.motion_frames.append(((x_min+x_max+y_min+y_max)/4,[x_min, y_min, x_max, y_max]))
            if len(VP.motion_frames) > 20:
                VP.motion_frames.pop(0)
            VP.in_motion = True
        if VP.in_motion:
            show_box = False
            if VP.check_movement((xmid, ymid)):
                VP.pos = (xmid, ymid)
                cv2.circle(frame, VP.pos,13,(0,255,0),4)
                VP.change_frame([int(x_min),int(y_min),int(x_max),int(y_max)])
                VP.still_counter = 0
            else:
                VP.still_counter += 1
            if VP.still_counter > 200:
                VP.in_motion = False
                VP.still_counter = 0
                VP.still = True
                VP.change_frame(VP.estimate_bed_frame())
                still_count = 0
                last_bed_frame = frame[int(VP.box_frame['U']):int(VP.box_frame['D']),int(VP.box_frame['L']):int(VP.box_frame['R'])]
        elif VP.still:
            show_box = True
            if still_count % 5 == 0:
                bed_frame = frame[int(VP.box_frame['U']):int(VP.box_frame['D']),int(VP.box_frame['L']):int(VP.box_frame['R'])]
                bed_frame_ratio = abs(np.mean(bed_frame) - np.mean(last_bed_frame))
                print(bed_frame_ratio)
                ys, xs = np.where(motion_mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    xmid = (x_min + x_max) // 2
                    ymid = (y_min + y_max) // 2
                    print('bfm',bed_frame_ratio)
                if bed_frame_ratio > 20:
                    print('potentiel')
                    VP.still = False
                    VP.potential_fall = True
                    if VP.check_potential_fall((xmid, ymid)):
                        print('chute_détectée')
                        #print(sdx)
                    else:
                        VP.still = True
                        VP.potential_fall = False
                still_count = 0
            else:
                still_count += 1













    #check_img = frame[check_frame[0][0]:check_frame[0][1],check_frame[1][0]:check_frame[1][1]]
    cv2.circle(frame, VP.pos, 10, (255, 0, 255), -1)
    cv2.circle(frame, (xmid, ymid), 1, (0, 0, 255), -1)
    if show_box:
        cv2.rectangle(frame,(int(VP.box_frame["L"]),int(VP.box_frame["U"])),(int(VP.box_frame["R"]),int(VP.box_frame["D"])),(0,255,0),2)

    cv2.putText(
        frame,  # image cible (ta sous-image)
        str(VP.still_counter),  # texte à écrire
        (10, 30),  # position du texte (x, y) dans le zoom
        cv2.FONT_HERSHEY_SIMPLEX,  # police
        0.7,  # taille de la police
        (0, 0, 255),  # couleur (rouge ici)
        2,  # épaisseur
        cv2.LINE_AA  # type de ligne, plus joli
    )
    cv2.imshow("preview", frame)
    #cv2.imshow('motion_mask',check_img )
    cv2.imshow("black and white",motion_mask)
    prev_gray = gray.copy()

    # --- Synchronisation temporelle pour vitesse correcte ---
    target_time = start_time + frame_idx * frame_duration
    now = time.time()
    wait_sec = target_time - now
    if wait_sec > 0:
        key = cv2.waitKey(int(wait_sec * 1000)) & 0xFF
    else:
        key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    frame_idx += 1

vc.release()
cv2.destroyAllWindows()
