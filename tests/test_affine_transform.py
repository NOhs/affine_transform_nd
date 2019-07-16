import mgen
import numpy as np

from affine_transform import transform


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
