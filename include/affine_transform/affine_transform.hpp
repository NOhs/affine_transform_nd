/*!
 *
 * \file affine_transform.hpp
 *
 * \brief File containing n-dimensional affine transformation functions
 *        that can be combined with various interpolation and boundary
 *        functions.
 *
 *
 */
#pragma once
#include <array>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "interpolation.hpp"

/*! \brief Namespace containing affine transform functionality
 *         for transforming n-dimnesional data.
 */
namespace affine_transform
{
/*! \brief Namespace containing implementation details for the affine
 *         transformation functionality.
 */
namespace detail
{
/*! \brief Inner For loop of the transform function.
 *
 *  Loops over the dimensions of the output image and fills it
 *  using the given interpolation and boundary functions to look
 *  up values in the given input image.
 *
 *  @param[in] point            The point from which to start to loop
 *                              in the input image coordinate system
 *  @param[in] input_image      The image from which to extract values
 *  @param[in,out] output_image The image to write to
 *  @param[in] dx               The vectors along which to iterate to
 *                              fill the output image
 *  @param[in,out] chunk        A temporary memory object which is used
 *                              to call the interpolation function
 *  @param[in] background_value A background value to use when accessing
 *                              points outside the input image. (might get
 *                              ignored by some boundary functions)
 *  @param[in] begin            The starting index of the first dimension
 *                              (can be used to parallelise the transform)
 *  @param[in] end              The end index of the first dimension
 *                              (can be used to parallelise the transform)
 *  @param[in] xs               Loop variables used to access the n-dimensional
 *                              data
 *
 *  @tparam Dim                 The dimensions of the images
 *  @tparam T                   The data type of the images
 *  @tparam Func                The interpolation order function
 *  @tparam BoundaryFunc        The boundary function to use
 *  @tparam Xs                  The loop indices, will be int
 */
template <int Dim, typename T, typename Func, typename BoundaryFunc,
          typename... Xs>
constexpr void transform_loop(
    Eigen::Matrix<double, Dim, 1> point,
    const pybind11::detail::unchecked_reference<T, Dim>& input_image,
    pybind11::detail::unchecked_mutable_reference<T, Dim>& output_image,
    const std::array<Eigen::Matrix<double, Dim, 1>, Dim>& dx,
    interpolation::Data<Func, Dim>& chunk, T background_value, int begin,
    int end, Xs... xs)
{
    for (int i = begin; i < end; ++i)
    {
        /* We are in the inner-most loop*/
        if constexpr (Dim == sizeof...(xs) + 1)
        {
            auto x_lower = std::array<int, Dim>{};
            auto x_relative = std::array<double, Dim>{};
            for (size_t l = 0; l < Dim; ++l)
            {
                x_lower[l] = point(l) - (point(l) < 0);
                x_relative[l] = point(l) - x_lower[l];
            }

            interpolation::extract<Func, BoundaryFunc, Dim>(
                chunk, input_image, x_lower, background_value);

            auto interpolate = [&chunk](auto... args) {
                return interpolation::apply_func(chunk, args...);
            };

            output_image(xs..., i) = std::apply(interpolate, x_relative);
        }
        /* Start more loops to iterate over the N-dimensional data*/
        else
        {
            transform_loop<Dim, T, Func, BoundaryFunc>(
                point, input_image, output_image, dx, chunk, background_value,
                0, output_image.shape(sizeof...(xs) + 1), xs..., i);
        }
        point += dx[sizeof...(xs)];
    }
}
}; // namespace detail

/*! \brief Extracts an image from a given one with a given
 *         coordinate system.
 *
 *  @param[in] origin            The origin of the coordinate system for the
 *                               resulting image in the coordinate system of
 *                               the input system.
 *  @param[in] dx                The vectors of the coordinate system of the
 *                               resulting image in the coordinate system of the
 *                               input system.
 *  @param[in] input_image       The image from which to extract data
 *  @param[in,out] output_image  The image in which to store the results
 *  @param[in] background_value  The value to use for points outside the input
 *                               image's domain. Might be ignored by some
 *                               boundary functions.
 *
 *  @tparam Dim                  The dimensionality of the image data
 *  @tparam T                    The datatype of the image data
 *  @tparam Func                 The interpolation order to use
 *  @tparam BoundaryFunc         The boundary function to use
 */
template <int Dim, typename T, template <typename> typename Func,
          typename BoundaryFunc>
void transform(const Eigen::Matrix<double, Dim, 1>& origin,
               const std::array<Eigen::Matrix<double, Dim, 1>, Dim>& dx,
               const pybind11::array_t<T>& input_image,
               pybind11::array_t<T>& output_image, T background_value)
{
    auto input = input_image.template unchecked<Dim>();
    auto output = output_image.template mutable_unchecked<Dim>();

#pragma omp parallel
    {
        typedef Func<T> _Func;

        interpolation::Data<_Func, Dim> chunk;

        int x_len = output.shape(0) / omp_get_num_threads();
        int x_start = x_len * omp_get_thread_num();
        int x_end = x_start + x_len;

        Eigen::Matrix<double, Dim, 1> local_origin = origin + x_start * dx[0];
        if (omp_get_thread_num() == omp_get_num_threads() - 1)
        {
            x_end = output.shape(0);
        }
        detail::transform_loop<Dim, T, _Func, BoundaryFunc>(
            local_origin, input, output, dx, chunk, background_value, x_start,
            x_end);
    }
}

}; // namespace affine_transform