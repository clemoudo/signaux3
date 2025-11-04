import cv2
import numpy as np
import time
data_file = 'data/chutes/chute_2.mp4'

Possible_fall = False

prev = time.time()
check_bed_data = ((445, 575), (115, 225))

cv2.namedWindow("preview")

vc = cv2.VideoCapture('chutes_retravaillées/chute_2.mp4')

# Read first frame to know the size
rval, frame = vc.read()
if not rval:
    exit()

# Resize ONCE to get target shape
target_w, target_h = 432, 768

frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

old_check_bed_frame = frame[
    check_bed_data[0][0]:check_bed_data[0][1],
    check_bed_data[1][0]:check_bed_data[1][1]
]

counter = 1

while True:
    rval, frame = vc.read()
    if not rval:
        break

    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    check_bed_frame = frame[
        check_bed_data[0][0]:check_bed_data[0][1],
        check_bed_data[1][0]:check_bed_data[1][1]
    ]

    if counter % 30 == 0:
        print("diff:", abs(np.mean(check_bed_frame)-np.mean(old_check_bed_frame)))
        old_check_bed_frame = check_bed_frame.copy()
        counter = 1
    else:
        counter += 1

    # Draw rectangle
    cv2.rectangle(frame, (150,445), (225,575), (255,0,0), 1)

    cv2.imshow("preview", frame)

    # <-- LE SEUL DELAI QUI DONNE LA VITESSE NORMALE
    if cv2.waitKey(1) == 27:
        break

vc.release()
cv2.destroyAllWindows()

