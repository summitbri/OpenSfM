import opensfm.reconstruction as orec
from opensfm.dataset_base import DataSetBase
from typing import Optional
import logging


logger: logging.Logger = logging.getLogger(__name__)

def run_dataset(dataset: DataSetBase, output: Optional[str]) -> None:
    """Rolling shutter correct a reconstructions.

    Args:
        input: input reconstruction JSON in the dataset
        output: input reconstruction JSON in the dataset
    """

    logger.info("Starting rolling shutter correction")

    reconstructions = dataset.load_reconstruction()
    camera_priors = dataset.load_camera_models()
    rig_cameras_priors = dataset.load_rig_cameras()
    tracks_manager = dataset.load_tracks_manager()

    # camera_id = camera_exif["camera"]
    # camera = camera_priors[camera_id]

    logger.info("Estimating camera velocities")
    for reconstruction in reconstructions:
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

            # Step 1. Compute camera velocities
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

                for cur_shot in all_shots[1:]:
                    cur_shot_time = exifs[cur_shot]['capture_time']

                    # Check that enough time has passed between shots
                    # in rare cases we cannot estimate velocity because time information is not granular
                    # enough (e.g. subsecond shot)
                    delta_time = cur_shot_time - prev_shot_time # seconds
                    if delta_time > 0:
                        cur_shot_origin = reconstruction.get_shot(cur_shot).pose.get_origin()

                        # Calculate velocity as m/s
                        exifs[cur_shot]['speed'] = list((cur_shot_origin - prev_shot_origin) / delta_time)

                        prev_shot = cur_shot
                        prev_shot_origin = cur_shot_origin
                        prev_shot_time = cur_shot_time
                    else:
                        exifs[cur_shot]['speed'] = [0.0, 0.0, 0.0]
                        logger.warning("Cannot compute velocity for %s (delta time 0)" % cur_shot)
            else:
                logger.info("Using EXIF data")
            
            for s in all_shots:
                logger.info("%s (%+.2f,%+.2f,%+.2f) m/s" % (s, *exifs[s]['speed']))




        #     reconstruction.add_correspondences_from_tracks_manager(tracks_manager)
        #     gcp = dataset.load_ground_control_points()
        #     orec.bundle(
        #         reconstruction, camera_priors, rig_cameras_priors, gcp, dataset.config
        #     )
    # dataset.save_reconstruction(reconstructions, output)
