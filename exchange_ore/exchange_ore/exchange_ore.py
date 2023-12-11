# exchange_ore.py

# common
import numpy as np
from ultralytics import YOLO
import cv2
# ros2 packages
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


# imgmsg => cv Mat => model => order => solvePnP => draw
class ImageProcessor:
    def __init__(self, model_path: str):
        self.model_ = YOLO(model=model_path)
        # for solvePnP
        self.length_ = 126.5 # mm
        self.object_points_ = np.array([
                                [126.5, 126.5, 0.],
                                [126.5, -126.5, 0.],
                                [-126.5, -126.5, 0.],
                                [-126.5, 126.5, 0.]
                                ])

        # self.camera_matrix_
        # self.distortion_matrix_
        # self.raw_image_
        # self.new_image_
        # self.predict_result_
        # self.rvec_
        # self.tvec_

    
    def GetImage(self, image: Image):
        self.raw_image_ = np.array(CvBridge().imgmsg_to_cv2(image, "bgr8: CV_8UC3")) # ros2 Image => opencv image(numpy array)
        self.new_image_ = self.raw_image_

    
    def ShowRawImage(self):
        cv2.imshow("Raw Image", self.raw_image_)


    def GetCameraInfo(self, camera_info: CameraInfo):
        self.camera_matrix_ = np.array(camera_info.k).reshape((3, 3))
        self.distortion_matrix_ = np.array(camera_info.d).reshape(5)


    def ModelPredict(self):
        self.predict_result_ = self.model_.predict(self.image_)[0]


    def ShowPredictImage(self):
        img_array = self.predict_result_.plot()
        # im_array = self.predict_result_.plot(kpt_radius=3, kpt_line=False, boxes=False, masks=False)
        img = Image.fromarray(img_array[..., ::-1])
        img = np.array(img, np.uint8)
        cv2.imshow("Predicted Image", img)


    # use GetCameraInfo, GetImage, ModelPredict first
    def SolvePnP(self):
        # the outermost point is always the first point, and the order is clockwise
        key_points = self.predict_result_.keypoints.numpy().xy # x-y coordinates of keypoints in numpy array form
        boxes = self.predict_result_.boxes.numpy()
        if len(boxes) == 4:
            imagePoints = key_points[:, 3] # innermost points
            is_success_1, imagePoints, center = self.Determine4PointsOrder(imagePoints, boxes.cls)

            if is_success_1:
                # visualize the order of the target points
                for i in range(len(imagePoints)):
                    cv2.putText(img=self.new_image_, text=str(i + 1), org=(int(imagePoints[i][0]), int(imagePoints[i][1])),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.8, color=(255, 255, 255))

                for i in range(len(imagePoints)):
                    cv2.circle(img=self.new_image_, center=(int(imagePoints[i][0]), int(imagePoints[i][1])), radius=3,
                                color=(0, 0, 255), thickness=1)

                cv2.circle(img=self.new_image_, center=(int(center[0]), int(center[1])), radius=3,
                            color=(0, 0, 255), thickness=1)

                is_success_2, self.rvec_, self.tvec_ = cv2.solvePnP(self.object_points_, imagePoints, self.camera_matrix_, self.distortion_matrix_)

                if not is_success_2:
                    print("\nWarning: ImageProcessor.SolvePnP(): solvePnP failed!\n")
                else: 
                    # visualize solvePnP result
                    real_center = np.zeros(3)
                    # normal vector: real center -> z_vec_point
                    # normal_vec_point = np.array([0, 0, 30])
                    xyz_vec_points = np.eye(3) * self.length_

                    real_center_pic = self.Object2Picture(real_center)
                    # normal_vec_point_pic = self.Object2Picture(normal_vec_point)

                    xyz_vec_points_pic = np.zeros((3, 2))
                    for i in range(3):
                        xyz_vec_points_pic[i] = self.Object2Picture(xyz_vec_points[i])

                    # cv2.arrowedLine(img=self.new_image_, pt1=real_center_pic.astype(int), pt2=normal_vec_point_pic.astype(int), color=(255, 0, 255), thickness=2)

                    for i in range(3):
                        cv2.arrowedLine(img=self.new_image_, pt1=real_center_pic.astype(int), pt2=xyz_vec_points_pic[i].astype(int), color=(255, 0, 255), thickness=2)


    def ShowNewImage(self):
        cv2.imshow("New Image", self.new_image_)

    
    def OutputRvecTvec(self):
        return self.rvec_, self.tvec_


    # private
    # brief: bring a 3d point from the object coordinate system to the picture(2d point)
    # param: point: 3d point of the object with shape(3,)
    def Object2Picture(self, point):
        pic_point = np.zeros(2)
        rotation_matrix = np.zeros((3, 3))

        cv2.Rodrigues(self.rvec_, rotation_matrix)

        print("Debug: in Object2Picture(): rotation_matrix = ", rotation_matrix)

        # object coordinate system => camera coordinate system
        camera_point_3d = (np.dot(rotation_matrix, point.reshape((3, 1))).reshape((3,)) + self.tvec_.reshape((3,))) # Note: must reshape
        
        # x:0; y:1; z:2; 
        if camera_point_3d[2] != 0:
            # normalized x and y
            x = camera_point_3d[0] / camera_point_3d[2]
            y = camera_point_3d[1] / camera_point_3d[2]
            # r^2
            r_2 = x * x + y * y
            # distorted x and y
            distorted_x = x * (1 + self.distortion_matrix_[0] * r_2 + self.distortion_matrix_[1] * r_2 * r_2 + self.distortion_matrix_[4] * pow(r_2, 3)) + 2 * self.distortion_matrix_[2] * x * y + self.distortion_matrix_[3] * (r_2 + 2 * x * x)
            distorted_y = y * (1 + self.distortion_matrix_[0] * r_2 + self.distortion_matrix_[1] * r_2 * r_2 + self.distortion_matrix_[4] * pow(r_2, 3)) + 2 * self.distortion_matrix_[3] * x * y + self.distortion_matrix_[2] * (r_2 + 2 * y * y)
            # in the picture x and y
            distorted_point_3d = np.array([distorted_x, distorted_y, 1])
            pic_point = np.dot(self.camera_matrix_, distorted_point_3d).reshape((3,))[:2]

        return pic_point


    # private
    # brief: determine the order of the four target points in the picture
    # detail: take the special point(with 2 gaps) as the first and push others clockwise
    # param: points: a numpy array of 4 key points generated from the model with shape(4, 2)
    # param: cls: a numpy array of the classes of the 4 key points with shape(4,)
    # return: is_success: boolean
    # return: ordered_points: a numpy array of the 4 key points in the expected order
    # return: center: the central point
    def Determine4PointsOrder(self, points, cls):
        is_success = False
        ordered_points = np.zeros((4, 2))
        center = np.average(points, axis=0) # shape(2,)

        # preprocess

        if len(points) != 4 | len(cls) != 4:
            print("ImageProcessor.Determine4PointsOrder(): the length of one argument is not 4\n")
            return is_success, ordered_points, center

        # find all different classes
        cls_set = set(np.copy(cls).flatten().tolist()) # np.ndarray => list => set
        if len(cls_set) != 2:
            print("ImageProcessor.Determine4PointsOrder(): more than 2 point classes\n")
            return is_success, ordered_points, center

        # a boolean numpy array with shape(4,)
        is_special_cls = (cls == list(cls_set)[0]) # set cannot be accessed through indices
        tmp_len = len(cls[is_special_cls]) 
        if tmp_len == 1:
            pass
        elif tmp_len == 3:
            is_special_cls = ~is_special_cls # negate
        else:
            print("ImageProcessor.Determine4PointsOrder(): more or less than one special point\n")
            return is_success, ordered_points, center

        # key algorithm

        # sort by angles with respect to the center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]).tolist() # shape(4,)
        # a dictionary designed for sorting by angles
        tmp_dict = {angles[i]: (tuple(points[i]), is_special_cls[i]) for i in range(len(points))} # dictionary: (angle: (point, bool)); np.ndarray(unhashable) => tuple
        angles.sort() # ascending

        sorted_points_bool = [tmp_dict[i] for i in angles]
        sorted_points = np.array([sorted_points_bool[i][0] for i in range(len(points))])
        sorted_is_special_cls = np.array([sorted_points_bool[i][1] for i in range(len(points))])

        # take the special point as the first, others clockwise
        special_index = np.arange(0, len(points), 1, dtype="int")[sorted_is_special_cls].item() # 0 ~ len(points)-1; np.ndarray => scalar
        ordered_points = np.concatenate((sorted_points[special_index:], sorted_points[:special_index]), axis=0) # input a tuple

        is_success = True

        return is_success, ordered_points, center


    def __del__(self):
        cv2.destroyAllWindows()


class ExchangeOreNode(Node):
    def __init__(self, name, model_path, is_camera_info_changing=False):
        super().__init__(name)
        self.get_logger().info(f"Start ExchangeOreNode {name}!")
        self.img_processor_ = ImageProcessor(model_path)

        self.is_cam_info_change_ = is_camera_info_changing
        self.is_get_cam_info_ = False

        self.img_subscription_ = self.create_subscription(Image, "image_raw", self.OnRecievingImage)


    def OnRecievingImage(self, image):
        # if not self.is_cam_info_change_:
        #     if not self.is_cam_info_change_:
        #         self.img_processor_.GetCameraInfo()
        #         self.is_get_cam_info_ = True
        # else:
        #     self.img_processor_.GetCameraInfo()

        self.img_processor_.GetImage(image)
        self.img_processor_.ModelPredict()
        self.img_processor_.SolvePnP()
        self.img_processor_.ShowNewImage()
        # TODO output images to rqt


def main(args=None):
    rclpy.init(args)
    model_path = "../best.pt"
    node = ExchangeOreNode("test", model_path)
    rclpy.spin(node)
    rclpy.shutdown()


# useful info:

# official api for Image
# std_msgs/msg/Header header
# uint32 height
# uint32 width
# string encoding
# uint8 is_bigendian
# uint32 step
# uint8[] data

# official api for CameraInfo
# std_msgs/msg/Header header
# uint32 height
# uint32 width
# string distortion_model
# double[] d # distortion matrix
# double[9] k # 3*3; camera matrix
# double[9] r # 3*3; for monocular camera r is Identity
# double[12] p # 3*4; for monocular camera p is k + [0;0;0]
# uint32 binning_x
# uint32 binning_y
# sensor_msgs/msg/RegionOfInterest roi