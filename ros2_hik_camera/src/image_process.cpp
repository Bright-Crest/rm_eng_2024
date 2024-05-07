// used for 2 objects and 12 key points whose order is specified
// When the model is changed, the following functions must be carefully checked and changed:
// ImageProcessor::SolvePnP()

#include "image_process.hpp"

namespace image_process
{
  const std::vector<cv::Point3f> ImageProcessor::object_points_{
      {LEN_A, LEN_B, 0.0f}, {LEN_A, LEN_A, 0.0f}, {LEN_B, LEN_A, 0.0f}, {-LEN_B, LEN_A, 0.0f}, {-LEN_A, LEN_A, 0.0f}, {-LEN_A, LEN_B, 0.0f}, {-LEN_A, -LEN_B, 0.0f}, {-LEN_A, -LEN_A, 0.0f}, {-LEN_B, -LEN_A, 0.0f}, {LEN_B, -LEN_A, 0.0f}, {LEN_A, -LEN_A, 0.0f}, {LEN_A, -LEN_B, 0.0f}};
  // the right plate
  const std::vector<cv::Point3f> ImageProcessor::side_plate_object_points_{
      {144.0f, SIDE_LEN_A, -SIDE_LEN_A - SIDE_LEN_B}, {144.0f, 0, SIDE_LEN_B}, {144.0f, -SIDE_LEN_A, -SIDE_LEN_A - SIDE_LEN_B}};

  bool ImageProcessor::DetermineAbitraryPointsOrder(const std::vector<cv::Point2f> &whole_points, const std::vector<std::string> &classes, std::vector<int> &order)
  {
    // 3 key points(whole points) corresponding to 1 bounding box(class)
    if (whole_points.size() / 3 != classes.size())
      return false;

    std::vector<cv::Point2f> points;
    for (size_t i = 0; i < classes.size(); ++i)
      points.emplace_back(whole_points[3 * i + 1]);

    int special_corner_count = std::count(classes.begin(), classes.end(), ExchangeInfo::Special_Corner_Tag);
    if (special_corner_count > 1)
    {
      std::cerr << "Error: ImageProcessor::DetermineAbitraryPointsOrder(): more than 1 special corner count\n";
      return false;
    }

    // calculate the angle between two line
    auto calcAngle = [](const cv::Point2f &line_start, const cv::Point2f &line_end) -> float
    {
      float norm_line_start = sqrt(line_start.dot(line_start));
      float norm_line_end = sqrt(line_end.dot(line_end));
      return line_start.dot(line_end) / (norm_line_end * norm_line_start);
    };

    int normal_corner_count = classes.size() - special_corner_count;
    if (normal_corner_count > 3)
    {
      std::cerr << "Error: ImageProcessor::DetermineAbitraryPointsOrder(): more than 3 normal corner count\n";
      return false;
    }

    // start to calculate the order of the corner
    float threshold = 0.08;
    std::vector<std::pair<float, int>> angles_and_indices{};
    order.clear();
    if (special_corner_count == 1)
    {
      int special_idx = std::distance(classes.begin(), std::find(classes.begin(), classes.end(), ExchangeInfo::Special_Corner_Tag));
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
      if (normal_corner_count == 3)
      {
        cv::Point2f center{0, 0};
        for (auto &point : points)
        {
          center.x += point.x;
          center.y += point.y;
        }
        center.x /= points.size();
        center.y /= points.size();
        for (auto &i : angles_and_indices)
          order.emplace_back(i.second);
        auto tmp_it = std::find(order.begin(), order.end(), special_idx);
        std::rotate(order.begin(), tmp_it, order.end());
      }
      else if (normal_corner_count == 2 || normal_corner_count == 1)
      // normal_corner_count = 1, 2
      {
        order = {-1, -1, -1, -1};
        cv::Point2f ref_line_1 = whole_points[3 * special_idx + 2] - whole_points[3 * special_idx + 1];
        cv::Point2f ref_line_2 = whole_points[3 * special_idx] - whole_points[3 * special_idx + 1];
        order.reserve(4);
        order[0] = special_idx;
        for (auto &pair : angles_and_indices)
        {
          const auto &pair_idx = pair.second;
          if (pair_idx == special_idx)
            continue;
          cv::Point2f pair_line = whole_points[3 * pair_idx + 2] - whole_points[3 * pair_idx + 1];
          float theta_1 = calcAngle(ref_line_1, pair_line);
          float theta_2 = calcAngle(ref_line_2, pair_line);
          /**     ref_line_1   ref_line_2
           * 1      ~ 0           ~ 1
           * 2      ~ -1          ~ 0
           * 3      ~ 0           ~ -1
           */
          if (abs(theta_1) > abs(theta_2) && abs(theta_1 + 1) < threshold)
            order[2] = pair_idx;
          else if (abs(theta_2 - 1) < threshold)
            order[1] = pair_idx;
          else if (abs(theta_2 + 1) < threshold)
            order[3] = pair_idx;
          else
            std::cerr << "Warning: ImageProcessor::DetermineAbitraryPointsOrder(): can not completely determine order\n";
        }
      }
    } // special_corner_count == 0
    else
    {
      if (normal_corner_count == 3)
      {
      }
      else
      {
        return false;
      }
    }
    return true;
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

  bool ImageProcessor::SolvePnP(const cv::Mat &frame)
  {
    image_points_.clear();
    std::vector<cv::Point3f> src_points{};

    std::vector<cv::Point2f> whole_required_points{};
    std::vector<std::string> classes{};
    std::vector<int> order{};

    for (auto &detection : predict_result_)
    {
      if (detection.confidence < 0.65)
        continue;
      // 1: outermost point
      whole_required_points.emplace_back(detection.keypoints[0]);
      whole_required_points.emplace_back(detection.keypoints[1]);
      whole_required_points.emplace_back(detection.keypoints[2]);
      classes.emplace_back(detection.class_name);
    }

    int special_corner_count = std::count(classes.begin(), classes.end(), ExchangeInfo::Special_Corner_Tag);
    int normal_corner_count = classes.size() - special_corner_count;
    if (special_corner_count == 1 && normal_corner_count >= 1)
    // the front plate situation
    {
      if (!DetermineAbitraryPointsOrder(whole_required_points, classes, order))
        return false;
      // apply order to points
      // the size of order must be 4
      for (int i = 0; i < 4; ++i)
      {
        const int &order_elem = order[i];
        if (order_elem == -1)
          continue;
        for (auto &key_point : predict_result_[order_elem].keypoints)
          image_points_.emplace_back(key_point);
        src_points.emplace_back(object_points_[i * 3]);
        src_points.emplace_back(object_points_[i * 3 + 1]);
        src_points.emplace_back(object_points_[i * 3 + 2]);
      }

      if (image_points_.size() != src_points.size())
      {
        std::cout << image_points_.size() << " " << src_points.size();
        return false;
      }
      if (image_points_.size() == 3)
      {
        for (int i = 0; i < 2; ++i)
        {
          src_points.emplace_back((src_points[i] + src_points[i + 1]) / 2.f);
          image_points_.emplace_back((image_points_[i] + image_points_[i + 1]) / 2.f);
        }
      }
    }
    else if (special_corner_count == 0 && normal_corner_count == 1)
    // the side plate situation
    // @todo add the left side situation
    {
      image_points_ = whole_required_points;
      src_points = side_plate_object_points_;
      for (int i = 0; i < 2; ++i)
      {
        src_points.emplace_back((src_points[i] + src_points[i + 1]) / 2.f);
        image_points_.emplace_back((image_points_[i] + image_points_[i + 1]) / 2.f);
      }
    }
    else
    {
      return false;
    }

    if (!cv::solvePnP(src_points, image_points_, camera_matrix_, distortion_coefficients_, rvec_, tvec_))
    {
      std::cerr << "Failed to solvePnP\n";
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

} // namespace image_process
