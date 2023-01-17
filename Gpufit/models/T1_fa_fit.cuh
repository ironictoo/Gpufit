#ifdef USE_T1_FA_EXPONENTIAL
#define GPUFIT_T1_FA_EXPONENTIAL_CUH_INCLUDED
#define PI 3.14159265359

__device__ void calculate_t1_fa_exponential(
    float const * parameters,
    int const n_fits,
    int const n_points,
    float * value,
    float * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
	// indices
	REAL* user_info_float = (REAL*)user_info;
	
	// split user_info array into dependent variables theta (in degrees) and tr
	REAL theta = 0;
	REAL tr = 0;
    theta = user_info_float[point_index] * PI/180;		// convert theta into radians for trig funcs
    tr = user_info_float[point_index + n_points];

	// below commented code block is basically if you want to define every single data point and such
	/*if (!user_info_float)
	    {
	        theta = point_index;
	        tr = point_index;
	    }
	    else if (user_info_size / sizeof(REAL) == n_points*2)
	    {
	        theta = user_info_float[point_index];
	        tr = user_info_float[point_index + n_points];
	    }
	    else if (user_info_size / sizeof(REAL) > n_points*2)
	    {
	        int const chunk_begin = chunk_index * n_fits * n_points;
	        int const fit_begin = fit_index * n_points;
	        theta = user_info_float[chunk_begin + fit_begin + point_index];
	        tr = user_info_float[chunk_begin + fit_begin + point_index + n_points];
	    }*/
	//REAL* theta = user_info_float;
	//REAL* tr = user_info_float + n_points;

	///////////////////////////// value //////////////////////////////
	// formula calculating fit model values
    value[point_index] = parameters[0] * ( (1 - exp(-tr/parameters[1])) * sin(theta)) / (1 - exp(-tr/parameters[1]) * cos(theta) );		// formula calculating fit model values
	// si = a * ( (1 - e^(-tr/t1)) * sin(theta)) / (1 - e^(-tr/t1) * cost(theta) )
	
    /////////////////////////// derivative ///////////////////////////
    float * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = ( (1 - exp(-tr/parameters[1])) * sin(theta)) / (1 - exp(-tr/parameters[1]) * cos(theta) );  // formula calculating derivative values with respect to parameters[0]
	current_derivative[1 * n_points] = (parameters[0] * tr * sin(theta) * exp(tr/parameters[1]) * (cos(theta) - 1)) / (pow(parameters[1],2) * pow(exp(tr/parameters[1])-cos(theta),2));	// formula calculating derivative values with respect to parameters[1]

 }	


#endif
