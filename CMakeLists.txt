#/*
# * Copyright (C) General Motors Company. All rights reserved.
# * This information is confidential and proprietary to GM Company and may not be used, modified, copied or distributed.
# */
cmake_minimum_required( VERSION 3.11.0 )
project(decision_making VERSION 0.1.0 LANGUAGES C CXX)


pack_python(src decision_making)
pack_python(test decision_making)
pack_python(resources decision_making)
pack_python_file(paths.py decision_making)
pack_python_file(__init__.py decision_making)

add_component_info("decision_making")