
# Applications

if( USE_CUBLAS )
    add_definitions( -DUSE_CUBLAS )
endif()

function( add_example modules name )
  set( target ${name} )
  add_executable( ${target} ${name}.cpp
    ${PROJECT_SOURCE_DIR}/tests/utils.h
    ${PROJECT_SOURCE_DIR}/tests/utils.cpp
  )  
  target_include_directories( ${target} PRIVATE ${PROJECT_SOURCE_DIR} )
  target_link_libraries( ${target} ${modules} )
  set_property( TARGET ${target}
    PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
  set_property( TARGET ${target} PROPERTY FOLDER GpufitCpufitExamples )
#  install( TARGETS ${target} RUNTIME DESTINATION bin )
endfunction()

add_example( "Cpufit;Gpufit" Gpufit_Cpufit_Performance_Comparison )

add_example( "Cpufit;Gpufit" Gpufit_Cpufit_Nvidia_Profiler_Test )

add_example( "Cpufit;Gpufit" Gpufit_Cpufit_Patlak )

add_example( "Cpufit;Gpufit" Patlak_Fitting_Comparison )

if (GPUFIT_USE_PATLAK AND GPUFIT_USE_TOFTS AND GPUFIT_USE_TOFTS_EXTENDED AND GPUFIT_USE_TISSUE_UPTAKE AND GPUFIT_USE_2CXM AND GPUFIT_USE_T1_FA_EXPONENTIAL)
    add_example( "Cpufit;Gpufit" Gpufit_Cpufit_MRI_Parity )
endif()
