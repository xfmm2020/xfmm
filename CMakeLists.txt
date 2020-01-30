cmake_minimum_required(VERSION 3.10)
project(cpp-tools)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(PkgConfig REQUIRED)

#c++ 扩展库 中的文件系统接口
find_path(BOOST_INCLUDE_DIR boost/cstdlib.hpp PATHS /usr/include)
include_directories(${BOOST_INCLUDE_DIR})
find_library(BOOST_FS_LIBRARIES boost_filesystem)
find_library(BOOST_SYS_LIBRARIES boost_system)
find_library(BOOST_RE_LIBRARIES boost_regex)
find_library(BOOST_PROGRAM_OPTIONS boost_program_options)
link_libraries(${BOOST_FS_LIBRARIES} ${BOOST_SYS_LIBRARIES} ${BOOST_RE_LIBRARIES} ${BOOST_PROGRAM_OPTIONS})

#opencv 库
pkg_check_modules(OPENCV REQUIRED opencv)
include_directories(${OPENCV_INCLUDE_DIRS})
link_libraries(${OPENCV_LIBRARIES})
add_compile_options(${OPENCV_CFLAGS_OTHER})

#设置install时的地址前缀为当前地址
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

#将可执行文件写入TEST_SOURCES中并循环执行编译安装过程
file(GLOB TEST_SOURCES ./*.cpp)
foreach(test_src ${TEST_SOURCES})
   get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    install(TARGETS ${test_name} DESTINATION bin)
endforeach(test_src)
