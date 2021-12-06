import time
import urllib.request
import cv2
import numpy as np

class MotionDetector:

    def __init__(self, delay=2):

        self.delay = delay

    def run(self):

        last_server_call = 0

        while True:
            flag, frame = self.motion_detection()
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

            # if any motion found send sms to the client
            if flag:
                current_time = time.time()

                cv2.imwrite(f'./images/motion_{str(current_time)}.png', frame)

                # send sms every 10 minutes
                if current_time - last_server_call > self.delay * 60:
                    # TODO send to server
                    last_server_call = current_time

    def motion_detection(self):

        # todo read images from esp32 camera
        frame1 = self.read_frame()
        frame2 = self.read_frame()

        diff = cv2.absdiff(frame1, frame2)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (5, 5))

        _, thresh = cv2.threshold(gray, 20, 250, cv2.THRESH_BINARY)
        # thresh = cv2.dilate(thresh, (5, 5))

        # using frame3 for drawing contours we can not use frame2 because gray and diff are based on frame2
        # so the contours will draw on them too
        frame3 = frame2.copy()
        contours, hierchay = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame3, contours, -1, (0, 255, 100), 2)

        # sorting contours because we want to draw the largest one:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # cv2.drawContours(frame3, contours, 0, (0, 255, 100), 2)

        if len(contours) > 0:
            print('MOTION')
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            # converting box to int
            box = np.int32(box)
            cv2.drawContours(frame3, [box], 0, (0, 255, 100), 2)

            return True, frame3

        return False, frame2

    def read_frame(self):

        # change the IP address below according to the
        # IP shown in the Serial monitor of Arduino code
        url = 'http://192.168.1.95/cam-lo.jpg'

        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        return frame


if __name__ == '__main__':
    detector = MotionDetector()
    detector.run()
