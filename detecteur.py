import cv2
import numpy as np
import time
import classes
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

data, fs = sf.read("son.mp3", dtype="float32")
real_time = True
if real_time:
    cam = 0
else:
    cam = "testcoco2.mp4"

donnes = []
donnes_ground = []
temps = []
cv2.namedWindow("preview")

vc = cv2.VideoCapture(cam)

person = []
potential_fall_frames = []
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

target_w, target_h = 1280,720
frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

counter = 1
potential_fall = False

#------------constante-------------
still_time_constant = 60
motion_treshold = 0.0005
motion_frame_length = 15
bed_frame_treshold = 9
under_bed_frame_treshold = 12
potential_fall_frames_length = 100
fall_frame_checking = 40
#------------------------


start_time = time.time()
frame_idx = 1

check_frame = ((0,500),(0,500))
fall_confirmed = False
last_bed_frame = None
last_uner_bed_frame = None
motion_traeshold = 0.0005
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # ou 'XVID' pour .avi
out = cv2.VideoWriter('sortie.mp4', fourcc, fps, (target_w, target_h))
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

    if not VP.still and not VP.potential_fall:
        motion_mask_ratio = np.count_nonzero(motion_mask) / motion_mask.size
    if counter % 20 != 0:
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
        if motion_mask_ratio > motion_treshold and not VP.potential_fall:
            VP.motion_frames.append(((x_min+x_max+y_min+y_max)/4,[x_min, y_min, x_max, y_max]))
            if len(VP.motion_frames) > motion_frame_length:
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
            if VP.still_counter > still_time_constant:
                VP.change_frame(VP.estimate_bed_frame())
                if abs(int(VP.box_frame['D'])-int(VP.box_frame['U'])) < abs(int(VP.box_frame['R'])-int(VP.box_frame['L']))/1.5:
                    VP.in_motion = False
                    still_count = 0
                    VP.still = True
                    last_bed_frame = frame[int(VP.box_frame['U']):int(VP.box_frame['D']),int(VP.box_frame['L']):int(VP.box_frame['R'])]
                    last_under_bed_frame = frame[int(VP.box_frame['D']):int(VP.box_frame['D']+(int(VP.box_frame['D']-int(VP.box_frame['U'])))),int(VP.box_frame['L']):int(VP.box_frame['R'])]
                VP.estimated_bed_frames.pop(-1)
                VP.still_counter = 0
        elif VP.still:
            show_box = True
            if still_count % 5 == 0:
                bed_frame = frame[int(VP.box_frame['U']):int(VP.box_frame['D']),int(VP.box_frame['L']):int(VP.box_frame['R'])]
                bed_frame_ratio = abs(np.mean(bed_frame) - np.mean(last_bed_frame))
                under_bed_frame = frame[int(VP.box_frame['D']):int(VP.box_frame['D']+(int(VP.box_frame['D']-int(VP.box_frame['U'])))),int(VP.box_frame['L']):int(VP.box_frame['R'])]
                under_bed_frame_ratio = abs(np.mean(under_bed_frame)-np.mean(last_under_bed_frame))
                donnes.append(bed_frame_ratio)
                donnes_ground.append(under_bed_frame_ratio)

                t = frame_idx / fps
                temps.append(t)
                if bed_frame_ratio > bed_frame_treshold and under_bed_frame_ratio >under_bed_frame_treshold:
                    ys, xs = np.where(motion_mask[int(VP.box_frame['U']):int(VP.box_frame['D']),int(VP.box_frame['L']):int(VP.box_frame['R'])] > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        x_min, x_max = xs.min(), xs.max()
                        y_min, y_max = ys.min(), ys.max()
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        xmid = (x_min + x_max) // 2
                        ymid = (y_min + y_max) // 2
                        print('bfm', bed_frame_ratio)
                    print('potentiel')
                    VP.still = False
                    if VP.check_movement((xmid, ymid)):
                        if VP.check_potential_fall((xmid, ymid)):
                            print('potentielle_chute_détectée')
                            potential_fall_frames = []
                            potential_fall_frames.append(motion_mask_ratio)
                            VP.potential_fall = True
                        else:
                            print('reset')
                            VP.potential_fall = False
                still_count = 0
            else:
                still_count += 1
        elif VP.potential_fall:
            print('top')
            motion_mask_ratio = np.count_nonzero(motion_mask) / motion_mask.size
            ys, xs = np.where(motion_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                xmid = (x_min + x_max) // 2
                ymid = (y_min + y_max) // 2
            potential_fall_frames.append(motion_mask_ratio)
            print(len(potential_fall_frames))
            print(potential_fall_frames)
            if len(potential_fall_frames) > potential_fall_frames_length:
                potential_fall_frames = potential_fall_frames[fall_frame_checking:]
                mavg = 0
                for f in potential_fall_frames:
                    mavg += f
                mavg = mavg / len(potential_fall_frames)
                print('mavg ',mavg)
                if mavg < 0.001:
                    print('chute confirmée')
                    plt.grid(True)
                    plt.show()
                    VP.still = False
                    VP.potential_fall = False
                    duree_sec = 3
                    fall_display_frames = int(duree_sec * fps)
                    fall_confirmed = True
                else:
                    VP.potential_fall = False
                    donnes.append(0)
                    donnes_ground.append(0)
                    t = frame_idx / fps
                    temps.append(t)

    if fall_confirmed and fall_display_frames > 0:
        cv2.putText(
            frame,
            "CHUTE CONFIRMEE",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )
        fall_display_frames -= 1
        if fall_display_frames <= 0 and fall_confirmed:
            sd.play(data,fs)
            fall_confirmed = False

    #check_img = frame[check_frame[0][0]:check_frame[0][1],check_frame[1][0]:check_frame[1][1]]
    cv2.circle(frame, VP.pos, 10, (255, 0, 255), -1)
    cv2.circle(frame, (xmid, ymid), 1, (0, 0, 255), -1)
    if show_box:
        cv2.rectangle(frame,(int(VP.box_frame["L"]),int(VP.box_frame["U"])),(int(VP.box_frame["R"]),int(VP.box_frame["D"])),(0,255,0),2)
        cv2.rectangle(frame,(int(VP.box_frame["L"]),int(VP.box_frame["D"])),(int(VP.box_frame["R"]),int(VP.box_frame["D"])+(int(VP.box_frame["D"])-int(VP.box_frame["U"]))),(255,0,0),2)
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
    out.write(frame)



plt.plot(temps, donnes)
plt.plot(temps,donnes_ground)
plt.xlabel('Temps (s)')
plt.ylabel('bfm (bed_frame_ratio)')
plt.title('Évolution de bfm en fonction du temps')
plt.grid(True)
plt.show()
vc.release()
out.release()
cv2.destroyAllWindows()
