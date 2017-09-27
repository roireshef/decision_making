cmake_minimum_required(VERSION 2.8.12)
project (decision_making)

set( REPOSITORY_HOME "." )
include( ${REPOSITORY_HOME}/buildDefaults.cmake )

pack_python(src decision_making)
pack_python(test decision_making)
pack_python(resources decision_making)

add_component_info("decision_making")