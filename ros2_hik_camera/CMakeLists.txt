cmake_minimum_required(VERSION 3.10)
project(hik_camera)

## Use C++14
## set(CMAKE_CXX_STANDARD 14)
## filesystem c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

## Cmake : hik_camera
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/hik_camera_node.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC hikSDK/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME} PUBLIC hikSDK/lib/amd64)
  install(
    DIRECTORY hikSDK/lib/amd64/
    DESTINATION lib
  )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME} PUBLIC hikSDK/lib/arm64)
  install(
    DIRECTORY hikSDK/lib/arm64/
    DESTINATION lib
  )
else()
  message(FATAL_ERROR "Unsupport host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_link_libraries(${PROJECT_NAME}
  FormatConversion
  MediaProcess
  MvCameraControl
  MVRender
  MvUsb3vTL
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN hik_camera::HikCameraNode
  EXECUTABLE ${PROJECT_NAME}_node
)

# only for setting the personal OpenCV 4.8 path
set(CMAKE_PREFIX_PATH ~/.local)
find_package(OpenCV 4.8 REQUIRED)

## Cmake : video_capturer
set(VIDEO_CAPTURER video_capturer)
ament_auto_add_library(${VIDEO_CAPTURER} SHARED 
  src/video_capturer.cpp
)
target_link_libraries(${VIDEO_CAPTURER}
  opencv_highgui  
)
rclcpp_components_register_node(${VIDEO_CAPTURER}
  PLUGIN video_capturer::VideoCapturerNode 
  EXECUTABLE ${VIDEO_CAPTURER}_node
)

# Cmake : image_process
# For serial function
 set(IMAGE_PROCESS image_process)
# ament_auto_add_library(${IMAGE_PROCESS} SHARED 
#   src/image_process.cpp
#   src/yolov8_inference/inference.cpp
#   src/serial/serial.cc
# )

# >>> for GPU inference >>>
# TODO: add_compile_definitions
option(ENABLE_GPU "Enable GPU inference if applicable" ON)
if (ENABLE_GPU MATCHES ON)
	MESSAGE("option ENABLE_GPU: ON")
	add_definitions(-DENABLE_GPU)
	set(TENSORRT_CPP_API tensorrt_cpp_api)
	set(YOLOV8_GPU yolov8_gpu)
	set(IMAGE_PROCESS image_process)

	ament_auto_add_library(${IMAGE_PROCESS} SHARED 
	  src/image_process.cpp
	  src/image_process_node.cpp
	  ${TENSORRT_CPP_API}/src/engine.cpp
	  src/yolov8_inference/gpu_inference.cpp
	  src/yolov8_inference/inference.cpp
	  src/serial/serial.cc
	)

	# # >>> tensorrt_cpp_api >>>
	# set(TENSORRT_CPP_API tensorrt_cpp_api)
	# 
	# # For finding FindTensorRT.cmake
	# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${TENSORRT_CPP_API}/cmake" ${CMAKE_MODULE_PATH})
	# 
	# # TODO: Specify the path to TensorRT root dir if using GPU Inference
	# if (NOT TensorRT_DIR)
	# set(TensorRT_DIR /usr/src/tensorrt)
	# endif()
	# # Use the correct version of CUDA
	# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
	# 
	# # Suppress some warnings of find_package CUDA 
	# if(POLICY CMP0146)
	# cmake_policy(SET CMP0146 OLD) 
	# endif()
	# 
	# find_package(TensorRT REQUIRED)
	# find_package(CUDA REQUIRED)
	# 
	# add_library(${TENSORRT_CPP_API} SHARED
	# ${TENSORRT_CPP_API}/src/engine.cpp
	# )
	# target_include_directories(${TENSORRT_CPP_API} PUBLIC ${TENSORRT_CPP_API}/include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
	# target_link_libraries(${TENSORRT_CPP_API} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
	# # <<< tensorrt_cpp_api <<<
	# 
	# # >>> yolov8 gpu inference >>>
	# set(YOLOV8_GPU yolov8_gpu)
	# 
	# add_library(${YOLOV8_GPU} SHARED
	# src/yolov8_inference/gpu_inference.cpp
	# )
	# target_link_libraries(${YOLOV8_GPU} ${TENSORRT_CPP_API} ${OpenCV_LIBS})
	# target_include_directories(${YOLOV8_GPU} PUBLIC include/yolov8_inference ${TENSORRT_CPP_API}/include)
	# # <<< yolov8 gpu inference <<<
	# 
	# target_link_libraries(${IMAGE_PROCESS} ${YOLOV8_GPU})

	# >>> tensorrt_cpp_api >>>
	# For finding FindTensorRT.cmake
	set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${TENSORRT_CPP_API}/cmake" ${CMAKE_MODULE_PATH})
	 
	# TODO: Specify the path to TensorRT root dir if using GPU Inference
	if (NOT TensorRT_DIR)
	  set(TensorRT_DIR /usr/src/tensorrt)
	endif()
	# Use the correct version of CUDA
	set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
	 
	find_package(TensorRT REQUIRED)
	find_package(CUDA REQUIRED)
	 
	target_include_directories(${IMAGE_PROCESS} PUBLIC ${TENSORRT_CPP_API}/include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
	target_link_libraries(${IMAGE_PROCESS} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
	# <<< tensorrt_cpp_api <<<
	
	# >>> yolov8 gpu inference >>>
	target_include_directories(${IMAGE_PROCESS} PUBLIC include/yolov8_inference)
	# <<< yolov8 gpu inference <<<
else()
	ament_auto_add_library(${IMAGE_PROCESS} SHARED 
	  src/image_process.cpp
	  src/image_process_node.cpp
	  src/yolov8_inference/inference.cpp
	  src/serial/serial.cc
	)
endif()
# <<< for GPU inference <<<

if(APPLE) # macOS
    find_library(IOKIT_LIBRARY IOKit)
    find_library(FOUNDATION_LIBRARY Foundation)
    target_sources(${IMAGE_PROCESS} PRIVATE
        src/serial/impl/unix.cc
        src/serial/impl/list_ports/list_ports_osx.cc
    )
     target_link_libraries(${IMAGE_PROCESS} ${FOUNDATION_LIBRARY} ${IOKIT_LIBRARY})
elseif(UNIX) # .*nix
    target_sources(${IMAGE_PROCESS} PRIVATE
        src/serial/impl/unix.cc
        src/serial/impl/list_ports/list_ports_linux.cc
     )
         target_link_libraries(${IMAGE_PROCESS} rt pthread)
elseif(WIN32) # Windows
    target_sources(${IMAGE_PROCESS} PRIVATE
        src/serial/impl/win.cc
        src/serial/impl/list_ports/list_ports_win.cc
    )
      target_link_libraries(${IMAGE_PROCESS} setupapi)
    ament_export_libraries(setupapi)
endif()

target_link_libraries(${IMAGE_PROCESS}
  opencv_highgui  
)


rclcpp_components_register_node(${IMAGE_PROCESS}
  PLUGIN image_process::ImageProcessNode 
  EXECUTABLE ${IMAGE_PROCESS}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_cpplint
    ament_cmake_uncrustify
  )
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  model
  config
  launch
)
