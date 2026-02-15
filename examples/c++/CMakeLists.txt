
function( add_example module name )
	add_executable( ${name} ${name}.cpp )
	target_link_libraries( ${name} ${module} )
	set_property( TARGET ${name}
		PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
	set_property( TARGET ${name} PROPERTY FOLDER GpufitExamples )
endfunction()

function( add_cuda_example module name )
	add_executable( ${name} ${name}.cu )
	target_link_libraries( ${name} ${module} )
	set_property( TARGET ${name}
		PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
	set_property( TARGET ${name} PROPERTY FOLDER GpufitExamples )
endfunction()

# Examples

if ( GPUFIT_USE_BASE_MODELS )
	add_example( Gpufit Simple_Example )
	add_example( Gpufit Linear_Regression_Example )
	add_example( Gpufit Exponential_Fitting )
endif()

if ( GPUFIT_USE_DIXON_3 )
	add_example( Gpufit Dixon_Fitting )
endif()

if ( GPUFIT_USE_GAUSSIAN2D )
	add_example( Gpufit Gauss_Fit_2D_Example )
endif()

if ( GPUFIT_USE_PATLAK )
	add_example( Gpufit Patlak_Fitting )
endif()

if ( GPUFIT_USE_TOFTS )
	add_example( Gpufit Tofts_Fitting )
endif()

if ( GPUFIT_USE_TOFTS_EXTENDED )
	add_example( Gpufit Tofts_Extended_Fitting )
endif()

if ( GPUFIT_USE_TISSUE_UPTAKE )
	add_example( Gpufit Tissue_Uptake_Fitting )
endif()

if ( GPUFIT_USE_2CXM )
	add_example( Gpufit 2CXM_Fitting )
endif()

if ( GPUFIT_USE_T1_FA_EXPONENTIAL )
	add_example( Gpufit T1_FA_Exponential_Fitting )
endif()

add_cuda_example( Gpufit CUDA_Interface_Example )
