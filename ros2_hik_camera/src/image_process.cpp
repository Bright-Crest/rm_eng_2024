// used for 2 objects and 12 key points whose order is specified
// When the model is changed, the following functions must be carefully checked and changed:
// ImageProcessor::SolvePnP()

#include "image_process.hpp"

namespace image_process
{
  template <typename _Tp>
  double CalcAngleOf2Vectors(const cv::Point_<_Tp> &vector1, const cv::Point_<_Tp> &vector2)
  {
      double norm_vector1 = std::sqrt(vector1.dot(vector1));
      double norm_vector2 = std::sqrt(vector2.dot(vector2));
      return vector1.dot(vector2) / (norm_vector1 * norm_vector2);
  }

  const std::vector<cv::Point3f> ImageProcessor::object_points_{
      {LEN_A, LEN_B, 0.0f}, {LEN_A, LEN_A, 0.0f}, {LEN_B, LEN_A, 0.0f}, {-LEN_B, LEN_A, 0.0f}, {-LEN_A, LEN_A, 0.0f}, {-LEN_A, LEN_B, 0.0f}, {-LEN_A, -LEN_B, 0.0f}, {-LEN_A, -LEN_A, 0.0f}, {-LEN_B, -LEN_A, 0.0f}, {LEN_B, -LEN_A, 0.0f}, {LEN_A, -LEN_A, 0.0f}, {LEN_A, -LEN_B, 0.0f}};
  // the right plate
  const std::vector<cv::Point3f> ImageProcessor::side_plate_object_points_{
      {144.0f, SIDE_LEN_A, -SIDE_LEN_A - SIDE_LEN_B}, {144.0f, 0, SIDE_LEN_B}, {144.0f, -SIDE_LEN_A, -SIDE_LEN_A - SIDE_LEN_B}};

  std::vector<yolov8::Detection> ImageProcessor::FilterObjects()
  {
    // do not modify predict_results_ in this method
    std::vector<yolov8::Detection> filtered_objects = predict_result_;

    // filter special/normal corners by confidences
    std::vector<yolov8::Detection> special_corners{};
    std::vector<yolov8::Detection> normal_corners{};
    for (auto &object : filtered_objects)
    {
      if (object.class_name == ExchangeInfo::Special_Corner_Tag)
        special_corners.emplace_back(object);
      else if (object.class_name == ExchangeInfo::Normal_Corner_Tag)
        normal_corners.emplace_back(object);
    }
    if (special_corners.size() > (long unsigned int)special_corner_num_)
    {
      std::sort(special_corners.begin(), special_corners.end(),
                [](const yolov8::Detection &a, const yolov8::Detection &b) -> bool
                { return a.confidence > b.confidence; });
      special_corners.resize(special_corner_num_);
    }
    if (normal_corners.size() > (long unsigned int)(expected_object_num_ - special_corner_num_))
    {
      std::sort(normal_corners.begin(), normal_corners.end(),
                [](const yolov8::Detection &a, const yolov8::Detection &b) -> bool
                { return a.confidence > b.confidence; });
      normal_corners.resize(expected_object_num_ - special_corner_num_);
    }

    std::cout << "special corners: " << special_corners.size() << "\t" << "normal corners: " << normal_corners.size() << std::endl;
    filtered_objects = std::move(normal_corners);
    for (auto &special_corner : special_corners)
    {
      filtered_objects.emplace_back(special_corner);
    }

    // filter by angles (only supports 3 currently)
    if (point_num_ == 3)
    {
      std::vector<std::vector<yolov8::Detection>::iterator> remove_iterators{};
      // filter by the angle of the 3 key points (not parrallel)
      for (auto it = filtered_objects.begin(); it != filtered_objects.end(); it++)
      {
        auto angle = CalcAngleOf2Vectors<float>(it->keypoints[2] - it->keypoints[1], it->keypoints[0] - it->keypoints[1]);
        if (std::abs(angle) <= PARALLEL_THRESHOLD) 
          remove_iterators.push_back(it);
      }
      for (auto &it : remove_iterators)
        filtered_objects.erase(it);
    }
    return filtered_objects;
  }

  bool ImageProcessor::DetermineAbitraryPointsOrder(const std::vector<cv::Point2f> &whole_points, const std::vector<std::string> &classes, std::vector<int> &order)
  {
    // only supports special_corner_num_ == 1
    if (special_corner_num_ != 1)
      return false;

    order.clear();
    order.reserve(expected_object_num_);
    unsigned int special_corner_count = std::count(classes.begin(), classes.end(), ExchangeInfo::Special_Corner_Tag);
    unsigned int normal_corner_count = classes.size() - special_corner_count;
    // get the first point of each object
    std::vector<cv::Point2f> points;
    for (unsigned int i = 0; i < classes.size(); ++i)
      points.emplace_back(whole_points[point_num_ * i]);

    // start to calculate the order of the corner by angles
    std::vector<std::pair<float, int>> angles_and_indices{};
    if (special_corner_count == special_corner_num_)
    {
      int special_idx = std::distance(classes.begin(), std::find(classes.begin(), classes.end(), ExchangeInfo::Special_Corner_Tag));

      // full normal corners
      if (normal_corner_count == expected_object_num_ - special_corner_num_)
      {
        // calculate angles (can be interpreted as the slopes with respect to the point of the special corner)
        for (unsigned int i = 1; i < points.size(); ++i)
        {
          int cur_idx = (i + special_idx) % points.size();
          float angle = std::atan2(points[cur_idx].y - points[special_idx].y, points[cur_idx].x - points[special_idx].x);
          angles_and_indices.emplace_back(std::make_pair(angle, cur_idx));
        }
        angles_and_indices.emplace_back(std::make_pair(0.f, special_idx));
        // sort in counterclockwise
        std::sort(angles_and_indices.begin(), angles_and_indices.end(),
                  [](const std::pair<float, int> &a, const std::pair<float, int> &b) -> bool
                  { return a.first >= b.first; });
        // get order by rotate
        for (auto &i : angles_and_indices)
          order.emplace_back(i.second);
        auto tmp_it = std::find(order.begin(), order.end(), special_idx);
        std::rotate(order.begin(), tmp_it, order.end());
      }
      else if (normal_corner_count > 0 && normal_corner_count < expected_object_num_ - special_corner_num_)
      // normal_corner_count = 1, 2
      {
        order.push_back(special_idx);
        for (unsigned int i = special_corner_num_; i < expected_object_num_ - special_corner_num_; i++)
          order.push_back(-1); // -1 stands for a missing corner
        cv::Point2f ref_line_1 = whole_points[3 * special_idx + 2] - whole_points[3 * special_idx + 1];
        cv::Point2f ref_line_2 = whole_points[3 * special_idx] - whole_points[3 * special_idx + 1];
        for (auto &pair : angles_and_indices)
        {
          const auto &pair_idx = pair.second;
          if (pair_idx == special_idx)
            continue;
          cv::Point2f pair_line = whole_points[3 * pair_idx + 2] - whole_points[3 * pair_idx + 1];
          float theta_1 = CalcAngleOf2Vectors<float>(ref_line_1, pair_line);
          float theta_2 = CalcAngleOf2Vectors<float>(ref_line_2, pair_line);
          /**     ref_line_1   ref_line_2
           * 1      ~ 0           ~ 1
           * 2      ~ -1          ~ 0
           * 3      ~ 0           ~ -1
           */
          if (abs(theta_1) > abs(theta_2) && abs(theta_1 + 1) < parallel_threshold_)
            order[2] = pair_idx;
          else if (abs(theta_1) < abs(theta_2) && abs(theta_2 - 1) < parallel_threshold_)
            order[1] = pair_idx;
          else if (abs(theta_1) < abs(theta_2) && abs(theta_2 + 1) < parallel_threshold_)
            order[3] = pair_idx;
          else
            std::cerr << "Warning: ImageProcessor::DetermineAbitraryPointsOrder(): can not completely determine order\n";
          // TODO: fix potential overriding
        }
      }
      else if (normal_corner_count == 0)
      {
        order.push_back(special_idx);
        for (unsigned int i = special_corner_num_; i < expected_object_num_ - special_corner_num_; i++)
          order.push_back(-1); // -1 stands for a missing corner
      }
      else // normal_corner_count > expected_object_num_ - special_corner_num_
      {
	      std::cerr << "normal_corner_count > expected_object_num_ - special_corner_num_" << std::endl;
        return false;
      }
    }
    else
    {
      // TODO: special_corner_count == 0
      std::cerr << "special_corner_count == 0" << std::endl;
      return false;
    }

    return true;
  }

  void ImageProcessor::init(double parallel_threshold, unsigned int expected_object_num, unsigned int special_corner_num)
  {
    parallel_threshold_ = parallel_threshold;
    expected_object_num_ = expected_object_num;
    special_corner_num_ = special_corner_num;

    if (special_corner_num_ != 1)
    {
      std::cerr << "Warning: ImageProcessor::init(): unsupported special corner number " << special_corner_num_ << " (only suppports 1 currently)\n";
    }
  }

  void ImageProcessor::GetCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info)
  {
    for (unsigned int i = 0; i < camera_info->k.size(); ++i)
      camera_matrix_(i / 3, i - i / 3 * 3) = camera_info->k[i];

    if (camera_info->d.size() != 5)
    {
      std::cerr << "Error: ImageProcessor::GetCameraInfo(): the distortion matrix size is not 5\n";
      throw std::exception();
    }

    for (unsigned int i = 0; i < camera_info->d.size(); ++i)
      distortion_coefficients_(i) = static_cast<float>(camera_info->d[i]);
  }

  void ImageProcessor::initFPStick()
  {
    last_system_tick_ = static_cast<double>(cv::getTickCount());
  }

  cv::Vec3f ImageProcessor::getTvec()
  {
    return tvec_;
  }

  cv::Vec3f ImageProcessor::getRvec()
  {
    return rvec_;
  }

  bool ImageProcessor::SolvePnP()
  {
    image_points_.clear();
    std::vector<cv::Point3f> src_points{};
    std::vector<cv::Point2f> whole_required_points{};
    std::vector<std::string> classes{};
    std::vector<int> order{};

    std::vector<yolov8::Detection> filtered_corners = FilterObjects();

    // preprocess; prepare points and classes for DetermineAbitraryPointsOrder()
    for (auto &detection : filtered_corners)
    {
      for (auto &point : detection.keypoints)
        whole_required_points.emplace_back(point);
      classes.emplace_back(detection.class_name);
    }

    unsigned int special_corner_count = std::count(classes.begin(), classes.end(), ExchangeInfo::Special_Corner_Tag);
    
    // only supports special_corner_num_ == 1
    if (special_corner_num_ != 1)
      return false;
    if (filtered_corners.size() > (long unsigned int)expected_object_num_)
    {
      return false;
    }
    if (special_corner_count > special_corner_num_)
    {
      return false;
    }

    // the front plate situation
    if (classes.size() >= 1)
    {
      if (!DetermineAbitraryPointsOrder(whole_required_points, classes, order))
      {
      	std::cerr << "Error: Failed to solvePnP\n";
        return false;
      }

      // apply order to points
      // the size of order must be expected_object_num_
      for (unsigned int i = 0; i < order.size(); ++i)
      {
        const int index = order[i];
        if (index == -1)
          continue;
        for (auto &key_point : filtered_corners[index].keypoints)
          image_points_.emplace_back(key_point);
        for (unsigned int j = 0; j < point_num_; j++)
          src_points.emplace_back(object_points_[point_num_ * i + j]);
      }

      // DEBUG; similar to `assert`
      if (image_points_.size() != src_points.size())
      {
        std::cerr << image_points_.size() << " " << src_points.size() << std::endl;
        return false;
      }

      // Insert middle points of the original points if only the points of one corner are left
      if (image_points_.size() == (long unsigned int)point_num_ * 1)
      {
        for (unsigned int i = 0; i < point_num_ - 1; ++i)
        {
          src_points.emplace_back((src_points[i] + src_points[i + 1]) / 2.f);
          image_points_.emplace_back((image_points_[i] + image_points_[i + 1]) / 2.f);
        }
      }
    }
    else
    {
      return false;
    }

    if (!cv::solvePnP(src_points, image_points_, camera_matrix_, distortion_coefficients_, rvec_, tvec_))
    {
      std::cerr << "Error: Failed to solvePnP\n";
      return false;
    }

    // from object frame to grasp frame
    cv::Matx33f rotation_matrix, last_rotation_matrix, temp_matrix;
    cv::Matx33f grasp2object = {-1, 0, 0, 0, 1, 0, 0, 0, -1};

    cv::Rodrigues(rvec_, rotation_matrix);
    rotation_matrix = rotation_matrix * grasp2object;

    // interpolation
    float interpolation_coeff = 0.75;
    cv::Vec3f interpolation_vec;
    cv::Rodrigues(last_rvec_, last_rotation_matrix);
    cv::transpose(last_rotation_matrix, temp_matrix);
    auto interpolation_matrix = temp_matrix * rotation_matrix;
    cv::Rodrigues(interpolation_matrix, interpolation_vec);
    cv::Rodrigues(interpolation_coeff * interpolation_vec, interpolation_matrix);
    cv::Rodrigues(last_rotation_matrix * interpolation_matrix, rvec_);
    last_rvec_ = rvec_;

    tvec_ = interpolation_coeff * tvec_ + (1.f - interpolation_coeff) * last_tvec_;
    last_tvec_ = tvec_;

    std::cout << "SolvePnP() success" << std::endl;
    return true;
  }

  cv::Point2f ImageProcessor::Object2Picture(const cv::Point3f &point)
  {
    cv::Matx33f rotation_matrix;
    cv::Rodrigues(rvec_, rotation_matrix);

    // object coordinate system => camera coordinate system
    cv::Vec3f camera_point_3d = rotation_matrix * point + cv::Point3f(tvec_);

    if (camera_point_3d[2] == 0)
      return cv::Point2f{};

    // normalized x and y
    float x = camera_point_3d[0] / camera_point_3d[2];
    float y = camera_point_3d[1] / camera_point_3d[2];
    // r^2
    float r_2 = x * x + y * y;

    const float &k1 = distortion_coefficients_[0];
    const float &k2 = distortion_coefficients_[1];
    const float &k3 = distortion_coefficients_[4];
    const float &p1 = distortion_coefficients_[2];
    const float &p2 = distortion_coefficients_[3];
    // distorted x and y
    float distorted_x = x * (1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * std::pow(r_2, 3)) + 2 * p1 * x * y + p2 * (r_2 + 2 * x * x);
    float distorted_y = y * (1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * std::pow(r_2, 3)) + 2 * p2 * x * y + p1 * (r_2 + 2 * y * y);
    cv::Point3f distorted_point_3d{std::move(distorted_x), std::move(distorted_y), 1};
    // x and y in the picture
    cv::Point3f tmp = camera_matrix_ * distorted_point_3d;

    return cv::Point2f{std::move(tmp.x), std::move(tmp.y)};
  }

  void ImageProcessor::AddPredictResult(cv::Mat img, bool add_boxes, bool add_keypoints, bool add_fps)
  {
    for (auto &detection : predict_result_)
    {
      if (add_boxes)
      {
        cv::rectangle(img, detection.box, cv::Scalar(255, 255, 255), 2);
        std::string classString = detection.class_name + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(detection.box.x, detection.box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(img, textBox, cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img, classString, cv::Point(detection.box.x + 5, detection.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
      }
      // prevent misidentification
      if (add_keypoints)
      {
        for (unsigned int i = 0; i < detection.keypoints.size(); ++i)
        {
          cv::circle(img, detection.keypoints[i], 3, cv::Scalar(255, 0, 255), 2);
        }
      }
    }
    if (add_fps)
    {
      double current_system_tick = static_cast<double>(cv::getTickCount());
      double inference_time = (current_system_tick - last_system_tick_) / cv::getTickFrequency() * 1000;
      last_system_tick_ = current_system_tick;
      int FPS = static_cast<int>(1000.0f / inference_time);
      cv::putText(img, std::to_string(FPS), cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2, 0);
    }
  }

  void ImageProcessor::AddImagePoints(cv::Mat img, bool add_order, bool add_points)
  {
    if (add_order)
    {
      for (unsigned int i = 0; i < image_points_.size(); ++i)
        cv::putText(img, std::to_string(i), image_points_[i], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2);
    }
    if (add_points)
    {
      for (auto &point : image_points_)
        cv::circle(img, point, 5, cv::Scalar(255, 0, 255), 3);
    }
  }

  void ImageProcessor::AddSolvePnPResult(cv::Mat img)
  {
      cv::Point3f real_center{};
      std::vector<cv::Point3f> xyz_points;
      for (unsigned int i = 0; i < 3; ++i)
      {
          cv::Vec3f tmp{};
          tmp(i) = LEN_A;
          xyz_points.emplace_back(std::move(tmp));
      }

      cv::Point2i real_center_pic = Object2Picture(real_center);
      for (auto &point : xyz_points)
      {
          cv::Point2i point_pic = Object2Picture(point);
          cv::arrowedLine(img, real_center_pic, point_pic, cv::Scalar(255, 0, 255), 2);
      }
  }
} // namespace image_process
