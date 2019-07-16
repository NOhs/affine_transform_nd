"""Module containing functionality for applying affine transformations to nd images."""

import numpy as np

from . import _affine_transform


def transform(
    input_image,
    linear_transformation,
    translation,
    order="linear",
    origin=None,
    output_image=None,
    background_value=0.0,
):
    """
    Transform an image using an affine transformation.

    This function applies an affine transformation to an image, meaning, it first
    applies a linear transformation (a combination of rotation, scaling. etc)
    after which additionally a translation can be included. For the linear transformation,
    the origin of the coordinate system can be freely chosen. E.g. you could rotate an
    image of two spheres around their common center of mass instead of rotating around
    the image center or the original origin of the image data ("The lower left corner").

    Furthermore, an option is available to choose how data is read from the
    given image, in case the affine transformation does not perfectly map pixels
    from the given image to the output image. This is the `order`` of the interpolation.

    Arguments
    ---------
    input_image : nd-array
        The data to transform
    linear_transformation : matrix
        A matrix of dimension ``(dim, dim)``, with ``dim`` being the dimension of the
        input data.
    translation : vector
        The translation part of the affine transformation
    order : {'linear', 'cubic'}
        The interpolation order to use for sampling the input image, default is ``'linear'``
    origin : vector, optional
        The origin to use for the linear transformation. By default, the center
        of the image is chosen
    output_image : nd-array, optional
        The image used for storing the results. If not set, memory will be
        allocated internally
    background_value : optional
        The background value to use in case points outside the input image
        are sampled

    Returns
    -------
    nd-image
        The given ``output_image`` or if not given a newly created array with the
        results

    Raises
    ------
    ValueError
        If the dimensions of the given inputs mismatch

    """
    # check dimensions
    if not all(v == input_image.ndim for v in linear_transformation.shape):
        raise ValueError(
            f"The given linear transformation is of shape {linear_transformation.shape}"
            f" but should be of shape ({input_image.ndim}, {input_image.ndim})"
            " for the given input image."
        )

    if not len(translation) == input_image.ndim:
        raise ValueError(
            f"The given translation has wrong dimensionality {len(translation)}"
            f" while the required dimensionality for the given input image is {input_image.ndim}."
        )
    else:
        translation = np.asarray(translation)

    if origin is None:
        origin = np.fromiter(
            ((x - 1) / 2 for x in input_image.shape), dtype=input_image.dtype
        )
    elif not len(origin) == input_image.ndim:
        raise ValueError(
            f"The given origin has wrong dimensionality {len(origin)}"
            f" while the required dimensionality for the given input image is {input_image.ndim}."
        )
    else:
        origin = np.asarray(origin)

    if output_image is None:
        output_image = np.zeros(input_image.shape, input_image.dtype)
    elif not input_image.shape == output_image.shape:
        raise ValueError(
            f"The given output image has shape {output_image.shape}, but needs to have the same"
            f" shape as the given input image with shape {input_image.shape}."
        )

    if order == "linear":
        transform = _affine_transform.transform_linear
    elif order == "cubic":
        transform = _affine_transform.transform_cubic
    else:
        raise ValueError(
            f'Order was given as "{order}". But only "cubic" and "linear" are valid options.'
        )

    # We transform the coordinate system, so we take the inverse
    linear_transformation = np.linalg.inv(linear_transformation)

    origin = (linear_transformation @ (-translation - origin)) + origin

    transform(
        origin,
        linear_transformation.T,  # columns
        input_image,
        output_image,
        background_value,
    )

    return output_image
