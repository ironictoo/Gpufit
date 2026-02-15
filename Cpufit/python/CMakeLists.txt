# Python Cpufit package

# Package files are always prepared so local development can use the staged tree.
set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/pyCpufit" )
set( setup_files
  "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg"
)
set( module_directory "${build_directory}/pycpufit" )
set( module_files
  "${CMAKE_CURRENT_SOURCE_DIR}/pycpufit/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/pycpufit/cpufit.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/pycpufit/version.py"
)
set( binary "$<TARGET_FILE:Cpufit>" )

add_custom_target( PYTHON_CPUFIT_PACKAGE
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${setup_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${module_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${module_files} ${module_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary} ${module_directory}
  COMMENT "Preparing pyCpufit package"
)
set_property( TARGET PYTHON_CPUFIT_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( PYTHON_CPUFIT_PACKAGE Cpufit )

if( NOT Python_FOUND )
  find_package( Python COMPONENTS Interpreter )
endif()

if( NOT Python_FOUND )
  message( STATUS "Python NOT found - skipping creation of Cpufit Python wheel!" )
  return()
endif()

add_custom_target( PYTHON_CPUFIT_WHEEL ALL
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${build_directory} "${Python_EXECUTABLE}" setup.py clean --all
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${build_directory} "${Python_EXECUTABLE}" setup.py bdist_wheel
  COMMENT "Building pyCpufit wheel"
)
set_property( TARGET PYTHON_CPUFIT_WHEEL PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( PYTHON_CPUFIT_WHEEL PYTHON_CPUFIT_PACKAGE )
