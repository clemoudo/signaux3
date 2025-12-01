from numpy.ma.extras import average




class vieux:
    def __init__(self,x,y,check_frames):
        self.pos = (x,y)
        self.box_frame = {"L":check_frames[0],"U":check_frames[1],"R":check_frames[2],"D":check_frames[3]}
        self.potential_fall = False
        self.still = False
        self.in_motion = False
        self.motion_frames = []
        self.still_counter = 0
        self.estimated_height = 0
        self.estimated_width = 0
        self.estimated_bed_frames = []

    def change_fall_state(self):
        self.potential_fall = not self.potential_fall


    def estimate_height_width(self):
        self.estimated_height = abs(self.box_frame['D'] - self.box_frame['U'])
        self.estimated_width = abs(self.box_frame['L'] - self.box_frame['R'])

    def check_potential_fall(self,pos):
        x_diff = abs(pos[0]-self.pos[0])
        y_diff = abs(pos[1]-self.pos[1])
        if y_diff > self.estimated_height:
            return True
        elif x_diff > self.estimated_width:
            return True
        else:
            return False

    def change_frame(self,check_frames):
        self.box_frame = {"L":check_frames[0],"U":check_frames[1],"R":check_frames[2],"D":check_frames[3]}

    def check_movement(self,pos):
        x_avg = (pos[0]+self.pos[0])//2
        y_avg = (pos[1]+self.pos[1])//2
        if abs(pos[0]-x_avg) > 50 or abs(pos[1]-y_avg) > 50:
            return True
        else:
            return False

    def estimate_bed_frame(self):
        L = 1000
        U = 1000
        R = 0
        D = 0
        for fr in self.motion_frames:
            if fr[1][0] < L:
                L = fr[1][0]
            if fr[1][1] < U:
                U = fr[1][1]
            if fr[1][2] > R:
                R = fr[1][2]
            if fr[1][3] > D:
                D = fr[1][3]
        L -= 0
        U -= 0
        R += 0
        D += 0
        self.estimated_bed_frames.append([L, U, R, D])
        if len(self.estimated_bed_frames) >20:
            self.estimated_bed_frames.pop(0)
        if len(self.estimated_bed_frames) >1:
            L = 0
            U = 0
            R = 0
            D = 0
            for fr in self.estimated_bed_frames:
                L += fr[0]
                U += fr[1]
                R += fr[2]
                D += fr[3]
                # i = not i
            L = L // len(self.estimated_bed_frames)
            U = U // len(self.estimated_bed_frames)
            R = R // len(self.estimated_bed_frames)
            D = D // len(self.estimated_bed_frames)
        return [L, U, R, D]
