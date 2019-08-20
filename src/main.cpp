/*!
 *
 * \file main.cpp
 *
 * \brief File containing the python bindings of the C++ affine transform
 *         functions.
 *
 *
 */
#include <regex>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "affine_transform/affine_transform.hpp"
#include "affine_transform/interpolation.hpp"

using namespace interpolation;
using namespace affine_transform;

/*! \brief Helper function to obtain function docstring.
 *
 *  @param[in] order   The order of the interpolation
 *  @param[in] bounds  The boundary function used
 *
 *  @returns A string containing a docstring for the affine
 *           transform function with the appropriate order and
 *           bounds.
 */
std::string get_function_description(std::string order, std::string bounds)
{
    std::string func_description = R"mydocdelimiter(
        Applies an affine transform using {ORDER} interpolation and {BOUNDS}
        boundary conditions.

        Note
        ----
        All inputs must of course have matching dimensions, e.g. for a 3D image
        your vectors must be of length 3.

        Arguments
        ---------
        origin : 1d-array
            The 0,0,0,... coordinate of the transformed image in the original
            image. E.g., if the image should move +5 in the x direction, the
            new origin would be at (-5,0,0,...) so that after +5 steps in x,
            the value in the original image at (0,0,0,...) is placed in the output
            image.
        dx : list of nd-arrays
            The transformed vectors along which to fill the new image. Note that
            a transformation of the coordinate system (used here) means that if you
            want to e.g. rotate your object in your image, the inverse rotation has to
            be applied to these unit vectors. For pure translations this would be
            the columns of the identity matrix.
        input_image : nd-array
            The array on which to apply the affine transform.
        output_image : nd-array
            The array used for storing the result.
        background_value
            The value to use when looking up values outside the given input image.
    )mydocdelimiter";

    auto order_regex = std::regex(R"(\{ORDER\})");
    auto bounds_regex = std::regex(R"(\{BOUNDS\})");

    return std::regex_replace(
        std::regex_replace(func_description, order_regex, order), bounds_regex,
        bounds);
}

/*! \brief Helper function to define affine transform functions
 *         in python of *up to* a given dimension.
 *
 *  @param[in] m            The python module for which to define the
 *                          functions
 *  @param[in] name         The name all the functions will share
 *  @param[in] description  The doc-string all functions will share
 *
 *  @tparam T               The image data_type
 *  @tparam Func            The interpolation order
 *  @tparam BoundaryFunc    The boundary function to use
 *  @tparam MaxDim          The maximum image dimension up to which to
 *                          register python functions
 */
template <typename T, template <typename> typename Func, typename BoundaryFunc,
          int MaxDim>
void register_affine_transform(pybind11::module& m, const char* name,
                               const char* description)
{
    if constexpr (MaxDim != 1)
    {
        register_affine_transform<T, Func, BoundaryFunc, MaxDim - 1>(
            m, name, description);
    }
    m.def(name, &transform<MaxDim, T, Func, BoundaryFunc>, description,
          pybind11::arg("origin"), pybind11::arg("dx"),
          pybind11::arg("input_image"), pybind11::arg("output_image"),
          pybind11::arg("background_value"));
}

/*! \brief Function that makes all defined interpolation functions visible to
 * Python.
 */
PYBIND11_MODULE(_affine_transform, transformation_module)
{
    constexpr auto MODULE_DESCRIPTION = R"mydocdelimiter(
            Module containing functions for affine transformation of nd-data.

            This module is a compiled C++ module. As such it has lots of
            overloaded functions that deal with the all the different data
            types it wants to support. In general the python module
            :any:`affine_transform` should be used which hides most of these
            implementation details.
        )mydocdelimiter";
    transformation_module.doc() = MODULE_DESCRIPTION;

    /* register cubic, constant up to dimension 5*/
    register_affine_transform<double, cubic, ConstantBoundary, 5>(
        transformation_module, "transform_cubic",
        get_function_description("cubic", "constant").c_str());

    register_affine_transform<float, cubic, ConstantBoundary, 5>(
        transformation_module, "transform_cubic",
        get_function_description("cubic", "constant").c_str());

    /* register linear, constant up to dimension 5*/
    register_affine_transform<double, linear, ConstantBoundary, 5>(
        transformation_module, "transform_linear",
        get_function_description("linear", "constant").c_str());

    register_affine_transform<float, linear, ConstantBoundary, 5>(
        transformation_module, "transform_linear",
        get_function_description("linear", "constant").c_str());
}