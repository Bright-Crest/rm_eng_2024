// used for 2 objects and 12 key points whose order is specified
// When the model is changed, the following functions must be carefully checked and changed:
// ImageProcessor::SolvePnP()

#include "image_process.hpp"

namespace image_process
{

  /// @details check classes and then take the special point(with 2 gaps) as the first and push others counterclockwise
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

    float threshold = 0.1;
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
      else // normal_corner_count = 0, 1, 2
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
      return false;
    }

    return true;
  }

  /// @details check classes and then take the special point(with 2 gaps) as the first and push others counterclockwise
  bool ImageProcessor::Determine4PointsOrder(const std::vector<cv::Point2f> &points, const std::vector<std::string> &classes, std::vector<int> &order)
  {
    // preprocess
    if (points.size() != 4)
      return false;
    if (classes.size() != 4)
      return false;

    // check classes; expected one special point and 3 other points of the same class

    // all class types
    std::set<std::string> classes_set{};
    for (const std::string &cls : classes)
      classes_set.emplace(cls);

    if (classes_set.size() != 2)
    {
      std::cerr << "Warning: ImageProcessor::Determine4PointsOrder(): more or less than 2 classes\n";
      return false;
    }

    int special_idx = 0;
    std::string special_class{};

    // get the special index
    int tmp_count = std::count(classes.begin(), classes.end(), *classes_set.begin());
    switch (tmp_count)
    {
    case 1:
      special_class = *classes_set.begin();
      break;
    case 3:
      special_class = *classes_set.rbegin();
      break;
    default:
      std::cout << "Warning: ImageProcessor::Determine4PointsOrder(): more or less than one key point is special class\n";
      return false;
    }
    for (unsigned int i = 0; i < classes.size(); i++)
    {
      if (classes[i] == special_class)
      {
        special_idx = i;
        break;
      }
    }

    // compute center
    cv::Point2f center{0, 0};
    for (auto &point : points)
    {
      center.x += point.x;
      center.y += point.y;
    }
    center.x /= points.size();
    center.y /= points.size();
    // determine order of the points

    // determine order by angles first
    order.clear();
    std::vector<std::pair<float, int>> angles_and_indices{};
    for (unsigned int i = 0; i < points.size(); ++i)
    {
      float angle = std::atan2(points[i].y - center.y, points[i].x - center.x);
      angles_and_indices.emplace_back(std::make_pair(angle, i));
    }

    // counterclockwise
    std::sort(angles_and_indices.begin(), angles_and_indices.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) -> bool
              { return a.first >= b.first; });

    for (auto &i : angles_and_indices)
      order.emplace_back(i.second);

    // determine final order by the special index
    auto tmp_it = std::find(order.begin(), order.end(), special_idx);
    std::rotate(order.begin(), tmp_it, order.end());

    return true;
  }

  ImageProcessor::ImageProcessor(const std::string &model_path, const cv::Size &model_shape,
                                 const float &model_score_threshold, const float &model_nms_threshold)
  {
    model_.init(model_path, model_shape, model_score_threshold, model_nms_threshold);
  }

  void ImageProcessor::GetCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info)
  {
    for (unsigned int i = 0; i < camera_info->k.size(); ++i)
    {
      camera_matrix_ << camera_info->k[i];
    }

    if (camera_info->d.size() != 5)
    {
      std::cerr << "Error: ImageProcessor::GetCameraInfo(): the distortion matrix size is not 5\n";
      throw std::exception();
    }

    for (unsigned int i = 0; i < camera_info->d.size(); ++i)
    {
      distortion_coefficients_ << camera_info->d[i];
    }
  }

  cv::Vec3f ImageProcessor::getTvec()
  {
    return tvec_;
  }

  cv::Vec3f ImageProcessor::getRvec()
  {
    return rvec_;
  }

  bool ImageProcessor::SolvePnP(int point_num)
  {
    image_points_.clear();
    std::vector<cv::Point3f> src_points{};

    std::vector<cv::Point2f> whole_required_points{};
    std::vector<std::string> classes{};
    std::vector<int> order{};

    for (auto &detection : predict_result_)
    {
      if (detection.confidence < 0.75)
        continue;
      // 1: outermost point
      whole_required_points.emplace_back(detection.keypoints[0]);
      whole_required_points.emplace_back(detection.keypoints[1]);
      whole_required_points.emplace_back(detection.keypoints[2]);
      classes.emplace_back(detection.class_name);
    }

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
    if (!cv::solvePnP(src_points, image_points_, camera_matrix_, distortion_coefficients_, rvec_, tvec_))
    {
      std::cerr << "Failed to solvePnP\n";
      return false;
    }

    return true;
  }

  cv::Point2f ImageProcessor::Object2Picture(const cv::Point3f &point)
  {
    cv::Matx33f rotation_matrix;
    cv::Rodrigues(rvec_, rotation_matrix);

    // cv::Mat identity_matrix = cv::Mat::eye(3, 3, CV_64F);
    // rotation_matrix = identity_matrix;
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

  void ImageProcessor::AddPredictResult(cv::Mat img, bool add_boxes, bool add_keypoints)
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
          // cv::putText(frame, std::to_string(i), detection.keypoints[i], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2);
        }
      }
    }
  }

  void ImageProcessor::AddImagePoints(cv::Mat img, bool add_order, bool add_points)
  {
    if (add_order)
    {
      for (unsigned int i = 0; i < image_points_.size(); ++i)
      {
        cv::putText(img, std::to_string(i), image_points_[i], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2);
      }
    }
    if (add_points)
    {
      for (auto &point : image_points_)
      {
        cv::circle(img, point, 5, cv::Scalar(255, 0, 255), 3);
      }
    }
  }

} // namespace image_process

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(image_process::ImageProcessNode)
