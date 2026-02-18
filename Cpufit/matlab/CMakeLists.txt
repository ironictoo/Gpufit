
# MATLAB Cpufit binding

if( NOT Matlab_FOUND )
  message( STATUS "No suitable Matlab and/or MX_Library version found - skipping Cpufit Matlab binding!" )
  return()
endif()

set( MATLAB_MEX_LINK_LIBS )
set( MATLAB_MEX_LINK_DIRS )
if( UNIX AND NOT APPLE )
  list( APPEND MATLAB_MEX_LINK_LIBS mex mx )
  if( Matlab_BINARIES_DIR )
    list( APPEND MATLAB_MEX_LINK_DIRS ${Matlab_BINARIES_DIR} )
  endif()
  if( Matlab_EXTERN_LIBRARY_DIR )
    list( APPEND MATLAB_MEX_LINK_DIRS ${Matlab_EXTERN_LIBRARY_DIR} )
  endif()
elseif( TARGET Matlab::mex AND TARGET Matlab::mx )
  list( APPEND MATLAB_MEX_LINK_LIBS Matlab::mex Matlab::mx )
else()
  list( APPEND MATLAB_MEX_LINK_LIBS ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY} )
endif()

# Matlab MEX FILE

set( Headers
  )

set( Sources
  mex/CpufitMex.cpp
  )

add_library( CpufitMex SHARED
  ${Headers}
  ${Sources}
  )
  
set_target_properties( CpufitMex
	PROPERTIES
		PREFIX ""
		SUFFIX .${Matlab_MEX_EXTENSION}
		RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

target_include_directories( CpufitMex PRIVATE ${Matlab_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
if( MATLAB_MEX_LINK_DIRS )
  target_link_directories( CpufitMex PRIVATE ${MATLAB_MEX_LINK_DIRS} )
endif()
target_link_libraries( CpufitMex Cpufit ${MATLAB_MEX_LINK_LIBS} )

if( UNIX AND NOT APPLE )
  set_target_properties( CpufitMex
    PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "\$ORIGIN"
      INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endif()

if( WIN32 )
  SET(CMAKE_SHARED_LINKER_FLAGS "/export:mexFunction")
endif()

add_matlab_launcher( CpufitMex "${CMAKE_CURRENT_SOURCE_DIR}" )

# MATLAB Cpufit + Gpufit PACKAGE

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/matlab" )
set( package_files
  "${CMAKE_CURRENT_SOURCE_DIR}/cpufit.m"
)
set( binary_gpufit $<TARGET_FILE:Cpufit> )
set( binary_mex $<TARGET_FILE:CpufitMex> )

add_custom_target( MATLAB_CPUFIT_GPUFIT_PACKAGE ALL
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${package_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_gpufit} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_mex} ${build_directory}	
  COMMENT "Adding Cpufit to Matlab package"
)
set_property( TARGET MATLAB_CPUFIT_GPUFIT_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( MATLAB_CPUFIT_GPUFIT_PACKAGE Cpufit CpufitMex )

# add launcher
