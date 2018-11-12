find_file(ROOT_APP buildDefaults.cmake ${REPOSITORY_HOME}/..)
	
if (ROOT_APP)
	# message(found root ${ROOT_APP})
	set(DEF_ROOT ${ROOT_APP})
	set(REPOSITORY_HOME ${REPOSITORY_HOME}/..)
	include( ${ROOT_APP})
	return()
else()
	message("-- no repository application found, please clone it from the project")
endif()



