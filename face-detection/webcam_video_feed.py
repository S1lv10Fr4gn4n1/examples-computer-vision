import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class WebcamVideoFeed(object):
    
    def __init__(self, refresh_interval, cam_index=0, frame_worker=None):
        self.refresh_interval = refresh_interval
        self.cam_index = cam_index
        self.frame_worker = frame_worker

    def __grab_frame(self, cap):
        _, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.frame_worker != None:
           return self.frame_worker(image)

        return image
    
    def __update_plot(self, i):
        self.imagePlot.set_data(self.__grab_frame(self.cap1))
        # im2.set_data(__grab_frame(cap2))

    def start(self):
        # initiate camera
        self.cap1 = cv2.VideoCapture(self.cam_index)
        # cap2 = cv2.VideoCapture(1)

        # create two subplots
        # ax1 = plt.subplot(1,2,1)
        # ax2 = plt.subplot(1,2,2)

        #create two image plots
        # im1 = ax1.imshow(__grab_frame(cap1))
        # im2 = ax2.imshow(__grab_frame(cap2))
        
        self.imagePlot = plt.imshow(self.__grab_frame(self.cap1))

        anim = animation.FuncAnimation(plt.gcf(), 
                                self.__update_plot, 
                                interval=self.refresh_interval)
        anim._start()

        plt.gcf().canvas.mpl_connect("key_press_event", self.__close)
        plt.show()

    def __close(self, event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
