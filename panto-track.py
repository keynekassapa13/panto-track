import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class BottomLine:
    '''
    BottomLine class
    Used as an object with parameters
    (for the sake of reading it easier)
    #BL = bottom line
    '''
    def __init__(self, status):
        self.bl = "Right"
        self.contour = None
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.combine = False
        self.combine_frame = 0

class Pantograph:
    '''
    Pantograph class
    Require filepath as parameter
    Pantograph point and the BL status is set from init
    '''
    def __init__(self, file):
        self.count = 0
        self.cap = cv2.VideoCapture(file)

        self.all_panto = (206, 233, 427, 99)
        self.left_panto = (207, 235, 108, 103)
        self.right_panto = (520, 233, 109, 98)

        self.bl_tp = (454, 207, 12, 25)

        _, self.frame = self.cap.read()
        self.bl = BottomLine("Right")
        self.crop = None
        # height = self.img.shape[0]
        self.f_width = self.frame.shape[1]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.bl_x = []
        self.bl_y = []
        self.graph = None
        self.ax.axis([0, self.f_width, 0, 1])

        self.tracker_allp = cv2.TrackerBoosting_create()

        # Do the work
        self.make_crop()
        self.make_contours()
        self.tracking()

        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def make_crop(self):
        '''
        Crop the area above pantograph
        '''
        self.crop = self.frame[
            5: int(self.all_panto[1]),
            int(self.all_panto[0]) : int(self.all_panto[0]) + int(self.all_panto[2])
        ]

    def make_contours(self):
        '''
        Contour the area above the pantograph
        Filtered area and decide the BL status based on the contours
        If the line is combined, then there is one contour detected only,
            this means that the BL will change Left <-> Right

        '''
        img = cv2.cvtColor(self.crop, cv2.COLOR_BGR2GRAY)
        img = (255-img)
        _, img = cv2.threshold(img, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
        # cv2.imshow('mask', img)

        _, contours, _ = cv2.findContours(
            img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cnt_dict = {}
        for cnt in contours:
            temp_area = cv2.contourArea(cnt)
            if (temp_area > 10):
                x, y, w, h = cv2.boundingRect(cnt)
                cnt_dict[x] = cnt

        all_x = [*cnt_dict]
        all_x.sort()
        print(all_x)

        if len(contours) == 1:
            self.bl.combine_frame = self.count
            self.bl.combine = True
            self.bl.contour = contours[0]
        elif len(contours) == 2 and self.bl.combine == True and self.bl.combine_frame + 5 <= self.count:
            if self.bl.bl == 'Right': self.bl.bl = 'Left'
            elif self.bl.bl == 'Left': self.bl.bl = 'Right'
            self.bl.combine = False

        if self.bl.bl == 'Right' and len(contours) >= 2:
            self.bl.contour = cnt_dict[all_x[-1]]
        elif self.bl.bl == 'Left' and len(contours) >= 2:
            self.bl.contour = cnt_dict[all_x[0]]

        self.bl.x, self.bl.y, self.bl.w, self.bl.h = cv2.boundingRect(self.bl.contour)
        self.make_graph(self.bl.x, self.count)

        cv2.drawContours(self.crop, [self.bl.contour], -1, (200, 100, 255), 2)

    def tracking(self):
        '''
        Track all the remaining frame in the video
        1. Track pantograph position
        1. Crop
        2. Contour
        3. Decide BL status
        '''
        ok = self.tracker_allp.init(self.frame, self.all_panto)

        while(self.cap.isOpened()):
            try:
                self.count += 1
                _, self.frame = self.cap.read()

                self.make_crop()
                self.make_contours()

                b_allp, self.all_panto = self.tracker_allp.update(self.frame)
                if b_allp:
                    p1 = (int(self.all_panto[0]), int(self.all_panto[1]))
                    p2 = (int(self.all_panto[0] + self.all_panto[2]), int(self.all_panto[1] + self.all_panto[3]))
                    cv2.rectangle(self.frame, p1, p2, (255,0,0), 2, 1)
                else:
                    cv2.putText(
                        frame,
                        "Tracking failure detected",
                        (100,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0,0,255),
                        2
                    )
                cv2.imshow('frame', self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def make_graph(self, x, y):
        '''
        Make graph real time as the frame goes using opencv2 window
        '''
        self.bl_x.append(self.all_panto[0] + x)
        self.bl_y.append(y)

        self.bl_x = self.bl_x[-200:]
        self.bl_y = self.bl_y[-200:]

        self.ax.clear()
        if len(self.bl_y) < 200:
            self.ax.axis([0, self.f_width, 0, 200])
            self.ax.plot(self.bl_x, self.bl_y)
        else:
            self.ax.axis([0, self.f_width, self.bl_y[0], self.bl_y[-1]])
            self.ax.plot(self.bl_x, self.bl_y)

        self.fig.canvas.draw()
        self.graph = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        self.graph  = self.graph.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.graph = cv2.cvtColor(self.graph,cv2.COLOR_RGB2BGR)
        cv2.imshow("Graph", self.graph)

Pantograph('./pantograph.avi')
