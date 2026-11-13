# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.ndimage: multidimensional image processing (filters, morphology,
# measurements, spline interpolation and Fourier-domain filters).

from ._filters import correlate1d, convolve1d, gaussian_filter1d, gaussian_filter, \
    prewitt, sobel, laplace, gaussian_laplace, gaussian_gradient_magnitude, \
    correlate, convolve, uniform_filter1d, uniform_filter, minimum_filter1d, \
    maximum_filter1d, minimum_filter, maximum_filter, rank_filter, median_filter, \
    percentile_filter, generic_filter1d, generic_filter, generic_laplace, \
    generic_gradient_magnitude, vectorized_filter
from ._morphology import generate_binary_structure, iterate_structure, binary_erosion, \
    binary_dilation, binary_opening, binary_closing, binary_propagation, \
    binary_fill_holes, binary_hit_or_miss, grey_erosion, grey_dilation, grey_opening, \
    grey_closing, morphological_gradient, morphological_laplace, white_tophat, \
    black_tophat, distance_transform_edt, distance_transform_cdt, distance_transform_bf
from ._measurements import label, find_objects, sum_labels, sum, mean, variance, \
    standard_deviation, minimum, maximum, median, minimum_position, maximum_position, \
    extrema, center_of_mass, histogram, labeled_comprehension, value_indices, watershed_ift
from ._interpolation import spline_filter1d, spline_filter, map_coordinates, \
    geometric_transform, shift, zoom, affine_transform, rotate
from ._fourier import fourier_gaussian, fourier_uniform, fourier_shift, fourier_ellipsoid
