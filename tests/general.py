import numpy as np
import os
import cv2 as cv
from datetime import datetime, timedelta

from dabstract.utils import *


def prepare_cctv_camera_data(folderpath):
    # random video create function
    def write_random_video(file_path, frames=25, fps=25, w=640, h=480):
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv.VideoWriter(file_path, fourcc, fps, (w, h))

        for k in range(frames):
            frame = np.random.randint(255, size=(h, w)).astype('uint8')
            frame = np.repeat(frame, 3, axis=1)
            frame = frame.reshape(h, w, 3)
            writer.write(frame)

        writer.release()

    # create files
    files = 20
    duration = 10
    fps = 25
    w = 320
    h = 240
    block_size = 15
    length = duration * fps
    tmp_length_lo = 0
    timestamp_now = datetime(day=9, month=6, year=2021, hour=9, minute=54, second=25, microsecond=391234)
    os.makedirs(folderpath, exist_ok=True)
    if not np.all([os.path.isfile(os.path.join(folderpath, '%s.mp4' % ((timestamp_now + k * duration * timedelta(seconds=1)).strftime('%Y-%m-%d_%H.%M.%S')))) for k in range(files)]):
        for k in range(files):
            tmp_frames = int((length + tmp_length_lo) // block_size) * block_size
            tmp_length_lo = (length + tmp_length_lo) - tmp_frames
            if not os.path.isfile(os.path.join(folderpath, '%d.mp4' % k)):
                write_random_video(os.path.join(folderpath,
                                                '%s.mp4' % (timestamp_now + k * duration * timedelta(seconds=1)).strftime('%Y-%m-%d_%H.%M.%S')),
                                   frames=int(tmp_frames),
                                   fps=fps,
                                   w=w,
                                   h=h)

def prepare_camera_data(folderpath):
        # random video create function
        def write_random_video(file_path, frames=25, fps=25, w=640, h=480):
            fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv.VideoWriter(file_path, fourcc, fps, (w, h))

            for k in range(frames):
                frame = np.random.randint(255, size=(h, w)).astype('uint8')
                frame = np.repeat(frame, 3, axis=1)
                frame = frame.reshape(h, w, 3)
                writer.write(frame)

            writer.release()

        # create files
        files = 20
        duration = 10
        fps = 25
        w = 320
        h = 240
        os.makedirs(folderpath, exist_ok=True)
        for k in range(files):
            if not os.path.isfile(os.path.join(folderpath, '%d.mp4' % k)):
                write_random_video(os.path.join(folderpath, '%d.mp4' % k),
                                   frames=int(duration * fps),
                                   fps=fps,
                                   w=w,
                                   h=h)


def prepare_anomaly_audio_data(folderpath):
    if not os.path.isdir(folderpath):
        from scipy.io.wavfile import write
        # normal class
        files = 20
        duration = 60
        sampling_rate = 16000
        subdb = 'normal'
        for k in range(files):
            os.makedirs(os.path.join(folderpath, subdb), exist_ok=True)
            write(os.path.join(folderpath, subdb, str(k) + '.wav'), sampling_rate,
                  0.1 * np.random.rand(duration * 16000))
        labels = np.zeros(files)
        np.save(os.path.join(folderpath, subdb + '_labels.npy'), labels)

        # abnormal class
        files = 20
        duration = 60
        sampling_rate = 16000
        subdb = 'abnormal'
        for k in range(files):
            os.makedirs(os.path.join(folderpath, subdb), exist_ok=True)
            write(os.path.join(folderpath, subdb, str(k) + '.wav'), sampling_rate,
                  np.random.rand(duration * 16000))
        labels = np.ones(files)
        np.save(os.path.join(folderpath, subdb + '_labels.npy'), labels)
