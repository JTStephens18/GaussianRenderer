# CMakeList.txt : CMake project for GaussianRenderer, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("GaussianRenderer")

# Add source to this project's executable.
add_executable (GaussianRenderer "GaussianRenderer.cpp" "GaussianRenderer.h" "Sphere.cpp" "Sphere.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET GaussianRenderer PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
find_package(PCL CONFIG REQUIRED)
target_link_libraries(GaussianRenderer PRIVATE ${PCL_LIBRARIES})

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(GaussianRenderer PRIVATE glfw)

find_package(GLEW REQUIRED)
target_link_libraries(GaussianRenderer PRIVATE GLEW::GLEW)

find_package(glm CONFIG REQUIRED)
target_link_libraries(GaussianRenderer PRIVATE glm::glm)