import mgen
import numpy as np
import pytest

from affine_transform import transform


def test_wrong_dimension_linear_transform():
    for dim in range(1, 6):
        image = np.ones((1,) * dim)
        for d in (x for x in range(1, 6) if x != dim):
            matrix = np.eye(5)
            with pytest.raises(ValueError):
                transform(image, matrix[:dim, :d], translation=(0,) * dim)
            with pytest.raises(ValueError):
                transform(image, matrix[:d, :dim], translation=(0,) * dim)
            with pytest.raises(ValueError):
                transform(image, matrix[:d, :d], translation=(0,) * dim)


def test_wrong_translation_dimension():
    for dim in range(1, 6):
        image = np.ones((1,) * dim)
        for d in (x for x in range(1, 6) if x != dim):
            with pytest.raises(ValueError):
                transform(image, np.eye(dim), translation=(0,) * d)


def test_wrong_origin_dimension():
    for dim in range(1, 6):
        image = np.ones((1,) * dim)
        for d in (x for x in range(1, 6) if x != dim):
            with pytest.raises(ValueError):
                transform(image, np.eye(dim), translation=(0,) * dim, origin=(0,) * d)


def test_wrong_output_image_dim():
    for dim in range(1, 6):
        image = np.ones((1,) * dim)
        for d in (x for x in range(1, 6) if x != dim):
            output = np.ones((1,) * d)
            with pytest.raises(ValueError):
                transform(
                    image, np.eye(dim), translation=(0,) * dim, output_image=output
                )


def test_wrong_output_image_origin_dim():
    for dim in range(1, 6):
        image = np.ones((1,) * dim)
        for d in (x for x in range(1, 6) if x != dim):
            with pytest.raises(ValueError):
                transform(
                    image,
                    np.eye(dim),
                    translation=(0,) * dim,
                    output_image_origin=(0,) * d,
                )


def test_wrong_order():
    image = np.ones((1,))
    with pytest.raises(ValueError):
        transform(image, np.eye(1), translation=(0,), order="fantastic")


def test_wrong_dtype():
    with pytest.raises(ValueError):
        transform(
            np.ones((1,)),
            np.eye(1),
            (0,),
            output_image=np.zeros((1,), dtype=np.float32),
        )


def test_data_type_conversion():
    image = np.ones((1,), dtype=int)
    output = transform(
        image, np.eye(1), translation=(0,), output_image_origin=(0.5,), order="linear"
    )
    assert output == 0.5


def test_float32():
    image = np.ones((1,), dtype=np.float32)
    output_image = np.zeros((1,), dtype=np.float32)
    output = transform(image, np.eye(1), (0,), output_image=output_image)
    assert output == 1
    assert output.dtype == np.float32
    assert output_image == 1


def test_different_input_argument_types():
    image = np.ones((1,), dtype=int)
    transform(image, np.eye(1, dtype=int), translation=(0,))


def test_smallest_image():
    for dim in range(1, 6):
        image = np.ones((1,) * dim)
        output = transform(image, np.eye(dim), translation=(0,) * dim)
        assert output == 1


def test_identity_transform():
    for image_size in range(2, 6):
        for dim in range(1, 6):
            image = np.ones((image_size,) * dim)
            output = transform(image, np.eye(dim), translation=(0,) * dim)
            np.testing.assert_allclose(output, image)


def test_non_square_shape():
    for image_size_x, image_size_y in ((x, 6 - x) for x in range(2, 6)):
        image = np.ones((image_size_x, image_size_y))
        output = transform(image, np.eye(2), translation=(0, 0))
        np.testing.assert_allclose(output, image)


def test_translation_by_whole_number():
    for dim in range(1, 6):
        image = np.ones((6,) * dim)
        for select_dim in range(dim):
            translation = [0] * dim
            translation[select_dim] = 2
            indices = [slice(None, None)] * dim
            indices[select_dim] = slice(2)
            expected_output = np.ones((6,) * dim)
            expected_output[tuple(indices)] = 0
            output = transform(image, np.eye(dim), translation=translation)
            np.testing.assert_allclose(output, expected_output)


def test_translation_by_third():
    for dim in range(1, 6):
        image = np.ones((6,) * dim) * 2
        for select_dim in range(dim):
            translation = [0] * dim
            translation[select_dim] = 2 / 3
            indices = [slice(None, None)] * dim
            indices[select_dim] = slice(1)
            expected_output = np.ones((6,) * dim) * 2
            expected_output[tuple(indices)] = 1 / 3 * 2
            output = transform(image, np.eye(dim), translation=translation)
            np.testing.assert_allclose(output, expected_output)


def test_background_value():
    for dim in range(1, 6):
        image = np.ones((6,) * dim) * 2
        for select_dim in range(dim):
            translation = [0] * dim
            translation[select_dim] = 2 / 3
            indices = [slice(None, None)] * dim
            indices[select_dim] = slice(1)
            expected_output = np.ones((6,) * dim) * 2
            expected_output[tuple(indices)] = 1 / 3 * 2 + 2 / 3 * 15
            output = transform(
                image, np.eye(dim), translation=translation, background_value=15.0
            )
            np.testing.assert_allclose(output, expected_output)


def test_rotation_multiple_90():
    for dim in range(2, 6):
        image_1 = np.ones((6,) * dim)
        image_2 = np.zeros((6,) * dim)
        image_2[(slice(None, 3),) * dim] = 1
        center_1 = None
        center_2 = (1,) * dim
        for image, center in zip((image_1, image_2), (center_1, center_2)):
            v1 = np.zeros(dim)
            v2 = np.zeros(dim)
            v1[-2] = 1
            v2[-1] = 1
            for angle in (np.pi / 2 * x for x in range(1, 4)):
                rotation = mgen.rotation_from_angle_and_plane(angle, v1, v2)
                output = transform(image, rotation, (0,) * dim, origin=center)
                # need some atol here since we are comparing exactly to 0
                np.testing.assert_allclose(output, image, atol=1e-10)


def test_cubic_interpolation():
    image = np.zeros((9,) * 1)
    image[3] = 13.3
    image_test = transform(image, np.eye(1), (1.7,), order="cubic")

    def catmull_rom_interp(p0, p1, p2, p3, x):
        return (
            (-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * (x ** 3)
            + (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3) * (x ** 2)
            + (-0.5 * p0 + 0.5 * p2) * x
            + p1
        )

    image_padded = np.pad(image, 3, mode="constant", constant_values=0)
    result_manual = np.zeros((9,))

    for i in range(9):
        result_manual[i] = catmull_rom_interp(*image_padded[i : i + 4], 0.3)

    np.testing.assert_allclose(image_test, result_manual, atol=1e-10)


def test_extract_slice():
    image = np.zeros((6,) * 3)
    image[(slice(3, 6),) * 3] = 1
    output_slice = np.empty((6, 6, 1))
    output = transform(
        image,
        np.eye(3),
        (0, 0, 0),
        output_image=output_slice,
        output_image_origin=(0, 0, 3),
    )

    np.testing.assert_allclose(output[:, :, 0], image[:, :, 3])
