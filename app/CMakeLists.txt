
add_executable(kitti_stereo kitti_stereo.cpp)
target_link_libraries(kitti_stereo stereoslam ${THIRD_PARTY_LIBS})