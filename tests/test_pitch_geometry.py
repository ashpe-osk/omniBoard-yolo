import sys
import unittest
from importlib import import_module
from importlib.util import find_spec
from unittest.mock import patch

from footy_cvision.core import BoundingBox, Detection
from footy_cvision.football.ball_trajectory import (
    BallTrajectory,
    BallTrajectoryConfig,
    select_highest_confidence_ball,
)
from footy_cvision.football.pitch_geometry import PitchGeometry, SoccerPitchConfig
from footy_cvision.pipeline.pipeline import Pipeline, PipelineConfig


class IdentityGeometry:
    def image_to_pitch(self, point):
        return (float(point[0]), float(point[1]))


@unittest.skipUnless(find_spec("numpy"), "NumPy is optional")
class PitchGeometryTests(unittest.TestCase):
    def setUp(self):
        self.landmarks = {
            "top_left": (10.0, 5.0),
            "top_right": (210.0, 5.0),
            "bottom_right": (210.0, 305.0),
            "bottom_left": (10.0, 305.0),
            "center": (110.0, 155.0),
        }
        self.image_keypoints = {
            "top_left": (0.0, 0.0),
            "top_right": (100.0, 0.0),
            "bottom_right": (100.0, 100.0),
            "bottom_left": (0.0, 100.0),
            "center": (50.0, 50.0),
        }

    def test_dlt_estimation_and_bidirectional_transform(self):
        import_module("numpy")

        config = SoccerPitchConfig(
            landmarks=self.landmarks,
            smoothing_alpha=1.0,
        )
        geometry = PitchGeometry(config)
        with patch.dict(sys.modules, {"cv2": None}):
            result = geometry.estimate(self.image_keypoints)

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "estimated")
        self.assertEqual(result.keypoint_count, 5)
        pitch = geometry.image_to_pitch((25.0, 40.0))
        self.assertIsNotNone(pitch)
        self.assertAlmostEqual(pitch[0], 60.0, places=6)
        self.assertAlmostEqual(pitch[1], 125.0, places=6)
        image = geometry.pitch_to_image(pitch)
        self.assertAlmostEqual(image[0], 25.0, places=6)
        self.assertAlmostEqual(image[1], 40.0, places=6)

    def test_reuses_last_valid_homography_only_within_limit(self):
        geometry = PitchGeometry(
            SoccerPitchConfig(
                landmarks=self.landmarks,
                smoothing_alpha=1.0,
                max_reuse_frames=1,
            )
        )
        with patch.dict(sys.modules, {"cv2": None}):
            estimated = geometry.estimate(self.image_keypoints)
            reused = geometry.estimate({})
            expired = geometry.estimate({})

        self.assertTrue(estimated.valid)
        self.assertTrue(reused.reused)
        self.assertEqual(reused.status, "reused")
        self.assertFalse(expired.reused)
        self.assertIsNone(expired.matrix)

    def test_ball_selector_and_pitch_space_outlier_output(self):
        balls = [
            Detection(BoundingBox(9, 9, 11, 11), 0.7, "ball"),
            Detection(BoundingBox(99, 99, 101, 101), 0.4, "ball"),
        ]
        self.assertIs(balls[0], select_highest_confidence_ball(balls))
        trajectory = BallTrajectory(
            BallTrajectoryConfig(max_speed_m_per_frame=5.0)
        )
        first = trajectory.process([balls[0]], IdentityGeometry(), 0)
        outlier = trajectory.process(
            [Detection(BoundingBox(99, 99, 101, 101), 0.9, "ball")],
            IdentityGeometry(),
            1,
        )
        self.assertTrue(first["valid"])
        self.assertFalse(outlier["valid"])
        self.assertEqual(outlier["reason"], "pitch_space_outlier")
        self.assertIsNone(outlier["pitch_position"])
        self.assertEqual(outlier["raw_pitch_position"], [100.0, 100.0])
        self.assertFalse(outlier["interpolated"])

    def test_pipeline_attaches_pitch_coordinates_and_trajectory(self):
        landmarks = {
            "top_left": (0.0, 0.0),
            "top_right": (100.0, 0.0),
            "bottom_right": (100.0, 100.0),
            "bottom_left": (0.0, 100.0),
        }
        geometry = PitchGeometry(
            SoccerPitchConfig(landmarks=landmarks, smoothing_alpha=1.0)
        )
        pipeline = Pipeline(
            config=PipelineConfig(
                pitch_geometry=geometry,
                ball_trajectory=BallTrajectoryConfig(
                    max_speed_m_per_frame=5.0
                ),
            )
        )
        result = pipeline.process_frame(
            {
                "pitch_keypoints": {
                    "top_left": (0.0, 0.0),
                    "top_right": (100.0, 0.0),
                    "bottom_right": (100.0, 100.0),
                    "bottom_left": (0.0, 100.0),
                },
                "detections": [
                    {
                        "bbox": [20, 20, 30, 40],
                        "confidence": 0.9,
                        "class_name": "player",
                    },
                    {
                        "bbox": [49, 49, 51, 51],
                        "confidence": 0.9,
                        "class_name": "ball",
                    },
                ],
            },
            0,
        )
        player = next(track for track in result.tracks if track.class_name == "player")
        self.assertAlmostEqual(player.metadata["pitch_position"][0], 25.0)
        self.assertAlmostEqual(player.metadata["pitch_position"][1], 40.0)
        self.assertEqual(player.metadata["homography_status"], "estimated")
        self.assertTrue(result.ball_trajectory["valid"])
        self.assertEqual(result.to_dict()["homography"]["status"], "estimated")


if __name__ == "__main__":
    unittest.main()
