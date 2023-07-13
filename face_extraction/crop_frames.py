from argparse import ArgumentParser
import os
import json
import cv2
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

from utils import get_video_paths

def crop_frames(video_path, box_folder, out_dir):
    name = os.path.basename(video_path).split('.')[0]

    dict = {}
    with open(os.path.join(box_folder, name + '.json'), 'r') as f:
        dict = json.load(f)

    os.mkdir(os.path.join(out_dir, name))

    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        success, frame = video.read()
        if not success:
            continue

        faces = dict[str(i)]
        for j, face in enumerate(faces):
            out_fol = os.path.join(out_dir, name, str(j))

            if not os.path.isdir(out_fol):
                os.mkdir(out_fol)
            if len(face) > 0:
                left = max(int(face[0]), 0)
                top = max(int(face[1]), 0)
                right = int(face[2])
                bottom = int(face[3])

                cropped = frame[top:bottom, left:right]
                cv2.imwrite(os.path.join(out_fol, str(i) + '.png'), cropped)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--videos_path', help='Path to videos\' root directory.')
    parser.add_argument('--boxes_path', help='Path to bounding boxes.')
    parser.add_argument('--output_path', help='Path to output directory.')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers.')

    options = parser.parse_args()

    video_paths = get_video_paths(base_path=options.videos_path)

    with Pool(processes=options.workers) as p:
        with tqdm(total=len(video_paths)) as pbar:
            for video in p.imap_unordered(partial(crop_frames, box_folder = options.boxes_path, out_dir = options.output_path), video_paths):
                pbar.update()