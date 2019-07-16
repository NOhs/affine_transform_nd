/*!
 *
 * \file interpolation.hpp
 *
 * \brief File containing n-dimensional interpolation functions
 *        for linear and cubic order.
 *
 *
 */
#pragma once
#include <array>
#include <cmath>
#include <utility>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

/*! \brief Namespace containing interpolation functionality for
 *         interpolating n-dimensional data using linear or cubic
 *         interpolation
 */
namespace interpolation
{
/*! \brief Struct for interpolation data
 *
 *  The Data struct is a simple wrapper of the std::array that allows
 *  you to have arrays of arrays. The idea is that for example for cubic
 *  interpolation, you need in 1D 4 data points, in 2D 4x4 data points etc.
 *  In 1D this array contains the necessary points for a given x. In 2D
 *  This array contains arrays for a given constant y whose interpolation
 *  results can then again be used for constant x, etc.
 *
 *  @tparam Func   The interpolation function for which to use the data.
 *                 This determines how many data points are needed. E.g.
 *                 cubic interpolation requires 4 points. The function must
 *                 provide the value NUMBER_OF_VALUES.
 *  @tparam Dim    The dimensionality of the data to interpolate. E.g. for
 *                 2 dimensional data, the result would be an array of arrays.
 */
template <typename Func, int Dim>
struct Data
{
    /*! \brief array of Dim-1 dimensional nested arrays.
     */
    std::array<Data<Func, Dim - 1>, Func::NUMBER_OF_VALUES> data;

    /*! \brief function to access this n-dimensional data
     *
     *  @param[in] index_0 the first index of the n-dimensional position
     *                     for which to look up the value
     *  @param[in] indices further indices depending on dimensionality,
     *                     i.e. Dim-1 indices for Dim-dimensional data.
     *
     *  @returns The value in the Dim-dimensional data structure at the
     *           given position
     *
     */
    template <typename... Indices>
    typename Func::VALUE_TYPE operator()(size_t index_0,
                                         Indices... indices) const
    {
        return data[index_0](indices...);
    }

    /*! \brief function to access this n-dimensional data
     *
     *  @param[in] index_0 the first index of the n-dimensional position
     *                     for which to look up the value
     *  @param[in] indices further indices depending on dimensionality,
     *                     i.e. Dim-1 indices for Dim-dimensional data.
     *
     *  @returns A reference to the value in the Dim-dimensional data structure
     *           at the given position
     *
     */
    template <typename... Indices>
    typename Func::VALUE_TYPE& operator()(size_t index_0, Indices... indices)
    {
        return data[index_0](indices...);
    }
};

/*! \brief 1D specialization of data
 *
 *  While all other Data<Func, Dim> structs contain arrays of Data<func,
 *  Dim-1>, the struct for 1D only contains an array of values of the
 *  value type the interpolation function works with. As such it works as
 *  a recursion stop.
 *
 *  @tparam Func   The interpolation function for which to use the data.
 *                 This determines how many data points are needed. E.g.
 *                 cubic interpolation requires 4 points.
 */
template <typename Func>
struct Data<Func, 1>
{
    /*! \brief Look-up type for nested arrays for the case of 1D.
     *
     *  In this special case of 1D the type is simply an array with the
     *  number of values and type defined by the interpolation function Func.
     *  E.g. for cubic<double> it would be std::array<double, 4>.
     */
    std::array<typename Func::VALUE_TYPE, Func::NUMBER_OF_VALUES> data;

    /*! \brief function to access this 1-dimensional data
     *
     *  @param[in] index the index of the 1-dimensional position for which to
     *                   look up the value
     *
     *  @returns The value in the 1-dimensional data structure at the
     *           given position
     *
     */
    typename Func::VALUE_TYPE operator()(size_t index) const
    {
        return data[index];
    }

    /*! \brief function to access this 1-dimensional data
     *
     *  @param[in] index the index of the 1-dimensional position for which to
     *                   look up the value
     *
     *  @returns A reference to the value in the 1-dimensional data structure at
     *           the given position
     *
     */
    typename Func::VALUE_TYPE& operator()(size_t index) { return data[index]; }
};

/*! \brief Cause a compile error for the illegal zero dimensional case
 */
template <typename Func>
struct Data<Func, 0>;

/*! \brief 1D linear interpolation
 *
 *  @tparam T   The datatype to use for the interpolation. E.g.
 *              float/double/etc.
 */
template <typename T>
struct linear
{
    /*! \brief The data type this function is operating with, e.g. for float
     *         images, it is float.
     */
    typedef T VALUE_TYPE;
    /*! \brief The number of datapoints required to perform the interpolation.
     *         Linear interpolation needs two datapoints.
     */
    static constexpr int NUMBER_OF_VALUES = 2;

    /*! \brief Compute the linear interpolation of the given points and
     * position.
     *
     *  Given 2 values, and a position in the interval [0, 1], this function
     *  returns the interpolated value using simple linear interpolation.
     *
     *  \note This function operates in double mode. The result is cast to the
     *        set template type T. NO rounding to nearest etc. is performed for
     *        e.g. integers.
     *
     *  @param[in] p   The 2 values to interpolate
     *  @param[in] x   The position in the interval [0, 1] to interpolate
     *
     *  @returns   The result of the interpolation
     */
    T operator()(const Data<linear<T>, 1>& p, double x)
    {
        return static_cast<T>(p(0) * (1 - x) + p(1) * x);
    }
};

/*! \brief 1D cubic interpolation
 *
 *  @tparam T   The datatype to use for the interpolation. E.g.
 *              float/double/etc.
 */
template <typename T>
struct cubic
{
    /*! \brief The data type this function is operating with, e.g. for float
     *         images, it is float.
     */
    typedef T VALUE_TYPE;
    /*! \brief The number of datapoints required to perform the interpolation.
     *         Cubic interpolation needs four datapoints.
     */
    static constexpr int NUMBER_OF_VALUES = 4;

    /*! \brief Compute the cubic interpolation of the given points and position.
     *
     *  Given 4 values, and a position in the interval [0, 1], this function
     *  returns the interpolated value using a uniform Catmull-Rom spline.
     *
     *  \note This function operates in double mode. The result is cast to the
     *        set template type T. NO rounding to nearest etc. is performed for
     *        e.g. integers.
     *
     *  @param[in] p   The 4 values to interpolate
     *  @param[in] x   The position in the interval [0, 1] to interpolate
     *
     *  @returns   The result of the interpolation
     */
    T operator()(const Data<cubic<T>, 1>& p, double x)
    {
        return static_cast<T>(
            p(1) + 0.5 * x *
                       (p(2) - p(0) +
                        x * (2.0 * p(0) - 5.0 * p(1) + 4.0 * p(2) - p(3) +
                             x * (3.0 * (p(1) - p(2)) + p(3) - p(0)))));
    }
};


/*! \brief Namespace containing implementation details for the interpolation
 *         functionality.
 */
namespace detail
{
/*! \brief Function to extract data from a pybind11 array using
 *         a std::array for indexing
 *
 *  Additional function required to realize the variadic indexing using
 *  the std::array.
 *
 *  @param[in] image          The image from which to extract a value
 *  @param[in] array_indices  The location of the data point to extract
 *
 *  @returns                  The value at the given position
 *
 *  @tparam T                 The data type of the image
 *  @tparam Dim               The dimensions of the image
 *  @tparam I                 Template parameter for unpacking the std::array
 */
template <typename T, int Dim, std::size_t... I>
T get_image_value_impl(
    const pybind11::detail::unchecked_reference<T, Dim>& image,
    const std::array<int, Dim>& array_indices, std::index_sequence<I...>)
{
    return image(array_indices[I]...);
}

/*! \brief Function to extract data from a pybind11 array using
 *         a std::array for indexing
 *  @param[in] image          The image from which to extract a value
 *  @param[in] array_indices  The location of the data point to extract
 *
 *  @returns                  The value at the given position
 *
 *  @tparam T                 The data type of the image
 *  @tparam Dim               The dimensions of the image
 */
template <typename T, int Dim>
T get_image_value(const pybind11::detail::unchecked_reference<T, Dim>& image,
                  const std::array<int, Dim>& array_indices)
{
    return get_image_value_impl<T, Dim>(image, array_indices,
                                std::make_index_sequence<Dim>{});
}

/*! \brief Apply Func to the given data and position.
 *
 * This function works by realising that the n-dimensional interpolation can
 * be broken down into a list of (n-1)-dimensional interpolations that are
 * interpolated via a 1D interpolation. This recursive call to
 * lower-dimensional interpolation is realized with recursive template
 * programming.
 *
 *  @param[in] indices   Indices used to unpack the array p to call the
 *                       lower dimensional interpolation on each sub-array
 *                       (for 2D and higher dimensional data).
 *  @param[in] chunk     N-dimensional array data for the interpolation.
 *                       Needs the right number of points in each dimension,
 *                       e.g. 4 for cubic interpolation. Should be nested
 *                       arrays in C order.
 *  @param[in] x         The first coordinate value of the N-dimensional
 *                       position at which to interpolate
 *  @param[in] xs        The last N-1 coordinate values of the N-dimensional
 *                       position. One argument for each dimension.
 *
 *  @returns             The interpolation value
 *
 *  @tparam Func         The interpolation order func
 *  @tparam Ts           The type of the position coordinate values. This is
 *                       enforced to be double
 *  @tparam I            Needed only to unpack array into separate
 *                       function calls.
 */
template <typename Func, typename... Ts, std::size_t... I>
static typename Func::VALUE_TYPE
apply_func_impl(std::index_sequence<I...> indices,
                const Data<Func, sizeof...(Ts) + 1>& chunk, double x, Ts... xs)
{
    if constexpr (sizeof...(Ts) > 0)
    {
        return apply_func_impl<Func>(
            indices,
            {{apply_func_impl(indices, std::get<I>(chunk.data), xs...)...}}, x);
    }
    else
    {
        return Func()(chunk, x);
    }
}

/*! \brief Extract interpolation patch from given data around given position.
 *
 *  This function uses variadic templates to iterate over n-dimensional
 *  data using the given boundary function.
 *
 *  @param[in,out] chunk           N-dimensional array data for the
 *                                 interpolation. Needs the right number of
 *                                 points in each dimension, e.g. 4 for cubic
 *                                 interpolation.
 *  @param[in] image               The image from which to extract the data.
 *  @param[in] lower_corner        Lower corner of the sub-cube to extract from
 *                                 the image into the given chunk.
 *  @param[in] background_value    The background value to use in case the value
 *                                 is outside the image domain (might be ignored
 *                                 depending on the boundary function used)
 *  @param[in] loop_indices        The loop indices of all the for loops to
 *                                 fill the n-dimensional chunk
 *
 *  @tparam Func                   The interpolation order func
 *  @tparam BoundaryFunc           The boundary type to use. E.g.
 *                                 ConstantBoundary
 *  @tparam Dim                    The dimensionality of the given image
 */
template <typename Func, typename BoundaryFunc, int Dim, typename... Ts>
static void extract_impl(
    Data<Func, Dim>& chunk,
    const pybind11::detail::unchecked_reference<typename Func::VALUE_TYPE, Dim>&
        image,
    const std::array<int, Dim>& lower_corner,
    typename Func::VALUE_TYPE background_value, Ts... loop_indices)
{
    for (int i = 0; i < Func::NUMBER_OF_VALUES; ++i)
    {
        if constexpr (Dim == sizeof...(Ts) + 1)
        {
            std::array<int, Dim> voxel_position{{loop_indices..., i}};

            for (int l = 0; l < Dim; ++l)
            {
                voxel_position[l] += lower_corner[l];
            }
            chunk(loop_indices..., i) =
                BoundaryFunc::template apply<typename Func::VALUE_TYPE, Dim>(image, voxel_position, background_value);
        }
        else
        {
            extract_impl<Func, BoundaryFunc, Dim>(chunk, image, lower_corner,
                                                  background_value,
                                                  loop_indices..., i);
        }
    }
}
}; // namespace detail

/*! \brief Struct for dealing with n-dimensional image boundaries by
 *         returning a constant values for coordinates out of bounds.
 */
struct ConstantBoundary
{
    /*! \brief Apply the constant boundary to the given image and position.
     *
     *  @param[in] image            The image to sample
     *  @param[in] voxel_position   The position at which to sample
     *  @param[in] background_value The background value to use if the given
     *                              position is out of bounds
     *
     *  @returns                    The value at the given position using the
     *                              rules defined by ConstantBoundary for
     *                              positions outside of the given image.
     *
     *  @tparam T                   The data type of the given image
     *  @tparam Dim                 The dimensions of the given image
     */
    template <typename T, int Dim>
    static T apply(const pybind11::detail::unchecked_reference<T, Dim>& image,
                   const std::array<int, Dim>& voxel_position,
                   T background_value)
    {
        for (int l = 0; l < Dim; ++l)
        {
            if (voxel_position[l] < 0 || voxel_position[l] >= image.shape(l))
            {
                return background_value;
            }
        }
        return detail::get_image_value<T, Dim>(image, voxel_position);
    }
};

/*! \brief Apply Func to the given data and position.
 *
 *  @param[in] chunk  N-dimensional array data for the interpolation. Needs
 *                    the right number of points in each dimension, e.g. 4
 *                    for cubic interpolation. Should be nested arrays in C
 *                    order.
 *  @param[in] x      The first coordinate value of the N-dimensional
 *                    position at which to interpolate
 *  @param[in] xs     The last N-1 coordinate values of the N-dimensional
 *                    position. One argument for each dimension.
 *
 *  @returns          The interpolation value
 *
 *  @tparam Func       The interpolation order func
 *  @tparam Ts        The type of the position coordinate values. This is
 *                    enforced to be double
 */
template <typename Func, typename... Ts>
static typename Func::VALUE_TYPE
apply_func(const Data<Func, sizeof...(Ts) + 1>& chunk, double x, Ts... xs)
{
    return detail::apply_func_impl(
        std::make_index_sequence<Func::NUMBER_OF_VALUES>{}, chunk, x, xs...);
}

/*! \brief Extract interpolation patch from given data around given position.
 *
 *  Different interpolation schemes require a different number of points
 *  to operate. This function takes the appropriate chunk and fills it with
 *  data based on the information of the function given.
 *
 *  @param[in,out] chunk           N-dimensional array data for the
 *                                 interpolation. Needs the right number of
 *                                 points in each dimension, e.g. 4 for cubic
 *                                 interpolation.
 *  @param[in] image               The image from which to extract the data.
 *  @param[in, out] point_floored  The point in the image where to interpolate
 *                                 floored. Will point to the lower corner of
 *                                 the data grid used for the interpolation at
 *                                 the end.
 *  @param[in] background_value    The background value to use in case the value
 *                                 is outside the image domain (might be ignored
 *                                 depending on the boundary function used)
 *  @tparam Func                    The interpolation order func
 *  @tparam BoundaryFunc           The boundary type to use. E.g.
 *                                 ConstantBoundary
 *  @tparam Dim                    The dimensionality of the given image
 */
template <typename Func, typename BoundaryFunc, int Dim>
static void
extract(Data<Func, Dim>& chunk,
        const pybind11::detail::unchecked_reference<typename Func::VALUE_TYPE,
                                                    Dim>& image,
        std::array<int, Dim>& point_floored,
        typename Func::VALUE_TYPE background_value)
{
    for (size_t l = 0; l < Dim; ++l)
    {
        point_floored[l] -= (Func::NUMBER_OF_VALUES - 2) / 2;
    }

    detail::extract_impl<Func, BoundaryFunc, Dim>(chunk, image, point_floored,
                                                  background_value);
}

}; // namespace interpolation
