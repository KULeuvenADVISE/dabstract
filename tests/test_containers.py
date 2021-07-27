import numpy as np
import os

from dabstract.dataprocessor.processors import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *

from general import prepare_anomaly_audio_data, prepare_camera_data, prepare_cctv_camera_data


def prepare_data():
    prepare_anomaly_audio_data(os.path.join('data', 'anomaly_audio'))
    prepare_camera_data(os.path.join('data', 'camera'))
    prepare_cctv_camera_data(os.path.join('data', 'cctv_camera'))


def test_MetaContainer():
    # ToDo
    pass


def test_FolderContainer():
    # ToDo
    pass


def test_CameraFolderContainer():
    from dabstract.dataset.containers import CameraFolderContainer

    # set up container
    container = CameraFolderContainer(os.path.join('data', 'camera'),
                                      map_fct=CameraDatareader(),
                                      extension='.mp4')

    # check data of container
    example, info = container.get(0, return_info=True)
    np.testing.assert_equal(info,
                            {'width': 320, 'height': 240, 'fs': 25.0, 'length': 250, 'duration': 10.0,
                             'output_shape': np.array([250, 320, 250])})
    assert example.shape == (250, 240, 320, 3)

    # check meta information
    np.testing.assert_equal(container.get_fs(), np.array([25., 25., 25., 25., 25., 25., 25., 25., 25., 25.,
                                                          25., 25., 25., 25., 25., 25., 25., 25., 25., 25.]))
    np.testing.assert_equal(container.get_samples(), np.array([250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
                                                               250, 250, 250, 250, 250, 250, 250, 250, 250, 250]))
    np.testing.assert_equal(container.get_duration(), np.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
                                                                10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]))
    np.testing.assert_equal(container.get_time_step(),
                            np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                      0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]))


def test_AsyncCameraFolderContainer():
    from dabstract.dataset.containers import AsyncCameraFolderContainer

    # set up container
    container = AsyncCameraFolderContainer(os.path.join('data', 'cctv_camera'),
                                           timestamp_ref_input='strftime',
                                           timestamp_new_input='minmax',
                                           timestamps_ref='%Y-%m-%d_%H.%M.%S.mp4',
                                           duration=2,
                                           error_margin=5,
                                           map_fct=CameraDatareader(),
                                           extension='.mp4')
    container[0]

    # check data of container
    example, info = container.get(0, return_info=True)
    np.testing.assert_equal(info,
                            {'width': 320, 'height': 240, 'fs': 25.0, 'length': 250, 'duration': 10.0,
                             'output_shape': np.array([250, 320, 250])})
    assert example.shape == (250, 240, 320, 3)

    # check meta information
    np.testing.assert_equal(container.get_fs(), np.array([25., 25., 25., 25., 25., 25., 25., 25., 25., 25.,
                                                          25., 25., 25., 25., 25., 25., 25., 25., 25., 25.]))
    np.testing.assert_equal(container.get_samples(), np.array([250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
                                                               250, 250, 250, 250, 250, 250, 250, 250, 250, 250]))
    np.testing.assert_equal(container.get_duration(), np.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
                                                                10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]))
    np.testing.assert_equal(container.get_time_step(),
                            np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                      0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]))


def test_AudioFolderContainer():
    # ToDo
    pass


def test_FeatureFolderContainer():
    # ToDo
    pass


if __name__ == "__main__":
    prepare_data()
    test_MetaContainer()
    test_FolderContainer()
    test_AudioFolderContainer()
    test_CameraFolderContainer()
    test_AsyncCameraFolderContainer()
    test_FeatureFolderContainer()
