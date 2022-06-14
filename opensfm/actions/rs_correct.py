import opensfm.reconstruction as orec
from opensfm.dataset_base import DataSetBase
from typing import Optional
import logging
import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

def run_dataset(dataset: DataSetBase, output: Optional[str]) -> None:
    """Rolling shutter correct a reconstructions.

    Args:
        output: output reconstruction JSON
    """

    logger.info("Starting rolling shutter correction")

    reconstructions = dataset.load_reconstruction()
    camera_priors = dataset.load_camera_models()
    rig_cameras_priors = dataset.load_rig_cameras()
    tracks_manager = dataset.load_tracks_manager()


    logger.info("Estimating camera velocities")
    for reconstruction in reconstructions:
        tracks_handler = orec.TrackHandlerTrackManager(tracks_manager, reconstruction)
        velocities = {}
        exifs = {}
        all_shots = []
        compute_speed = False

        for shot in reconstruction.shots:
            all_shots.append(shot)
            exifs[shot] = dataset.load_exif(shot)

            if not 'speed' in exifs[shot]:
                compute_speed = True
            
        # Sort by capture time, tie sorted by filename
        all_shots.sort(key=lambda s: (exifs[s]['capture_time'], s.lower()))

        if len(all_shots) > 0:

            # Compute camera velocities
            # - If we have speed values, we use those
            # - If we don't have speed values, we compute them

            if compute_speed:
                logger.info("Computing from camera poses")

                # Assume first image is stationary as most mission planning
                # software starts taking shots from stationary position
                prev_shot = all_shots[0]
                exifs[prev_shot]['speed'] = [0.0, 0.0, 0.0]
                prev_shot_origin = reconstruction.get_shot(prev_shot).pose.get_origin()
                prev_shot_time = exifs[prev_shot]['capture_time'] # seconds
                prev_camera_id = exifs[prev_shot]['camera']

                for cur_shot in all_shots[1:]:
                    cur_shot_time = exifs[cur_shot]['capture_time']
                    cur_camera_id = exifs[cur_shot]['camera']
                    cur_shot_origin = reconstruction.get_shot(cur_shot).pose.get_origin()

                    # Check that enough time has passed between shots
                    # in rare cases we cannot estimate velocity because time information is not granular
                    # enough (e.g. subsecond shot)
                    delta_time = cur_shot_time - prev_shot_time # seconds
                    
                    if cur_camera_id != prev_camera_id:
                        # Change of camera, assume 0
                        exifs[cur_shot]['speed'] = np.array([0.0, 0.0, 0.0])
                    elif delta_time > 0:
                        # Calculate velocity as m/s
                        exifs[cur_shot]['speed'] = (cur_shot_origin - prev_shot_origin) / delta_time
                    else:
                        exifs[cur_shot]['speed'] = np.array([0.0, 0.0, 0.0])
                        logger.warning("Cannot compute velocity for %s (delta time 0)" % cur_shot)
                    
                    prev_shot = cur_shot
                    prev_shot_origin = cur_shot_origin
                    prev_shot_time = cur_shot_time
                    prev_camera_id = cur_camera_id
            else:
                logger.info("Using EXIF data")
                for shot in exifs:
                    exifs[shot]['speed'] = np.array(exifs[shot]['speed'])

            for s in all_shots:
                logger.info("%s (%+.2f,%+.2f,%+.2f) m/s" % (s, *exifs[s]['speed']))


            for track_id in reconstruction.points:
                obs = tracks_handler.get_observations(track_id)
                coordinates = reconstruction.points[track_id].coordinates # 3D

                for shot_id in obs:
                    shot = reconstruction.shots[shot_id]
                    point = obs[shot_id].point # Original 2D observation

                    # Reproject 3D point to camera to get reprojected observation
                    reprojected_point = shot.project(coordinates)

                    error = reprojected_point - point

                    # Calculate new camera pose using the formula
                    # used in: A two-step approach for the correction of rolling
                    # shutter distortion in UAV photogrammetry
                    # https://www.sciencedirect.com/science/article/abs/pii/S0924271619302849

                    pixel_y = shot.camera.normalized_to_pixel_coordinates(point)[1]
                    rolling_shutter_readout = exifs[shot_id].get('rolling_shutter', 0) / 1000.0
                    origin = shot.pose.get_origin()

                    new_origin = origin + exifs[shot_id]['speed'] * rolling_shutter_readout * (pixel_y - shot.camera.height / 2.0) / shot.camera.height
                    
                    # Reproject the point using the new origin
                    shot.pose.set_origin(new_origin)
                    adjusted_reprojected_point = shot.project(coordinates)
                    
                    # Restore previous pose
                    shot.pose.set_origin(origin)

                    corrected_point = adjusted_reprojected_point + error

                    
                    print(corrected_point, point)
                    exit(1)
                exit(1)


        #     reconstruction.add_correspondences_from_tracks_manager(tracks_manager)
        #     gcp = dataset.load_ground_control_points()
        #     orec.bundle(
        #         reconstruction, camera_priors, rig_cameras_priors, gcp, dataset.config
        #     )
    # dataset.save_reconstruction(reconstructions, output)
