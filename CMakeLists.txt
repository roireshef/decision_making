cmake_minimum_required(VERSION 3.5.0)

project(decision_making VERSION ${CMAKE_UC_DECISION_MAKING_VERSION} DESCRIPTION "UC decision_making")

include(Uc_PythonUtils)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(srclink ${CMAKE_CURRENT_SOURCE_DIR}/*)
set(targetlink ${CMAKE_PYTHON_DIR}/decision_making)

ADD_CUSTOM_COMMAND(OUTPUT link_dir 
                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${targetlink}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${targetlink}
                    COMMAND cp -as ${srclink} ${targetlink} COMMENT "Generating symbolic link ${srclink}->${targetlink}"
                    )
ADD_CUSTOM_TARGET(decision_making DEPENDS link_dir)

add_dependencies(decision_making GROUP-common_data GROUP-rte)