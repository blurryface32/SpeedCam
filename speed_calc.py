import cv2
import math
import time
import numpy as np
import os

# Speed limit in km/hr
speed_limit = 80

# Folder name for storing traffic records
traffic_record_folder_name = "TrafficRecord"

if not os.path.exists(traffic_record_folder_name):
    os.makedirs(traffic_record_folder_name)
    os.makedirs(traffic_record_folder_name + "//exceeded")

# File location for recording speed
speed_record_file_location = traffic_record_folder_name + "//SpeedRecord.txt"
file = open(speed_record_file_location, "w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        self.id_count = 0
        self.et = 0
        self.s1 = np.zeros((1, 1000))
        self.s2 = np.zeros((1, 1000))
        self.s = np.zeros((1, 1000))
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0

    def update(self, objects_rect, ht):
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Check if the object is already detected
            same_object_detected = False

            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 70:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True

                    # Start timer
                    if (y >= int(0.625 * ht) and y <= int(0.75 * ht)):
                        self.s1[0, obj_id] = time.time()

                    # Stop timer and find difference
                    if (y >= int(0.25 * ht) and y <= int(0.375 * ht)):
                        self.s2[0, obj_id] = time.time()
                        self.s[0, obj_id] = self.s2[0, obj_id] - self.s1[0, obj_id]

                    # Capture flag
                    if (y < int(0.25 * ht)):
                        self.f[obj_id] = 1

            # New object detection
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.s[0, self.id_count] = 0
                self.s1[0, self.id_count] = 0
                self.s2[0, self.id_count] = 0

        # Assign new ID to objects
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # Calculate speed
    def get_speed(self, obj_id, dist):
        if (self.s[0, obj_id] != 0):
            speed = dist / self.s[0, obj_id]
        else:
            speed = 0

        return int(speed)

    # Save vehicle data
    def capture(self, img, x, y, h, w, speed, obj_id):
        if (self.capf[obj_id] == 0):
            self.capf[obj_id] = 1
            self.f[obj_id] = 0
            crop_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
            n = str(obj_id) + "_speed_" + str(speed)
            file = traffic_record_folder_name + '//' + n + '.jpg'
            cv2.imwrite(file, crop_img)
            self.count += 1
            filet = open(speed_record_file_location, "a")
            if (speed > speed_limit):
                file2 = traffic_record_folder_name + '//exceeded//' + n + '.jpg'
                cv2.imwrite(file2, crop_img)
                filet.write(str(obj_id) + " \t " + str(speed) + "<---exceeded\n")
                self.exceeded += 1
            else:
                filet.write(str(obj_id) + " \t " + str(speed) + "\n")
            filet.close()

    # Get speed limit
    def get_speed_limit(self):
        return speed_limit

    # Write summary to text file
    def end(self):
        file = open(speed_record_file_location, "a")
        file.write("\n-------------\n")
        file.write("-------------\n")
        file.write("SUMMARY\n")
        file.write("-------------\n")
        file.write("Total Vehicles :\t" + str(self.count) + "\n")
        file.write("Exceeded speed limit :\t" + str(self.exceeded))
        file.close()
