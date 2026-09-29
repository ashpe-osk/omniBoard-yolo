import unittest

from footy_cvision.core import BoundingBox, Detection
from footy_cvision.detection.fusion import ModelAwareFusion
from footy_cvision.football.possession import PossessionTracker
from footy_cvision.visualization.annotations import (
    TEAM_COLORS,
    draw_debug_frame,
    draw_football_ellipse,
    draw_tracking_frame,
)


class ModelAwareFusionTests(unittest.TestCase):
    def test_people_use_primary_while_ball_uses_specialist_with_fallback(self) -> None:
        primary = [
            Detection(BoundingBox(0, 0, 20, 40), 0.8, "Player"),
            Detection(BoundingBox(30, 0, 50, 40), 0.7, "Ball"),
        ]
        specialist = [
            Detection(BoundingBox(1, 0, 21, 40), 0.9, "PLAYER"),
            Detection(BoundingBox(31, 0, 36, 5), 0.6, "Ball"),
        ]
        fusion = ModelAwareFusion()
        fused = fusion({"primary": primary, "ball_referee": specialist})
        self.assertEqual([item.class_name for item in fused], ["player", "ball"])
        self.assertEqual(fused[0].metadata["model_source"], "primary")
        self.assertEqual(fused[1].metadata["model_source"], "ball_referee")

    def test_specialist_people_are_not_added_without_primary_people(self) -> None:
        specialist = [Detection(BoundingBox(0, 0, 20, 40), 0.95, "Player")]
        fused = ModelAwareFusion()({"ball_referee": specialist})
        self.assertEqual(fused, [])


class FootballAnnotationTests(unittest.TestCase):
    def test_team_color_feature_falls_back_when_torso_is_all_green(self) -> None:
        import numpy as np

        from footy_cvision.football.team_classification import (
            extract_team_color_feature,
        )

        frame = np.zeros((160, 120, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)
        feature = extract_team_color_feature(
            frame,
            BoundingBox(30, 10, 90, 150),
        )
        self.assertIsNotNone(feature)
        self.assertEqual(len(feature), 3)
        self.assertGreater(feature[1], 0.9)

    def test_team_classifier_maps_red_and_white_kits_to_stable_ids(self) -> None:
        import numpy as np

        from footy_cvision.football.team_classification import (
            TeamClassifier,
            extract_team_color_feature,
        )

        red_frame = np.zeros((160, 120, 3), dtype=np.uint8)
        red_frame[:] = (0, 0, 220)
        white_frame = np.zeros((160, 120, 3), dtype=np.uint8)
        white_frame[:] = (220, 220, 220)
        bbox = BoundingBox(20, 5, 100, 155)
        red = extract_team_color_feature(red_frame, bbox)
        white = extract_team_color_feature(white_frame, bbox)
        self.assertIsNotNone(red)
        self.assertIsNotNone(white)
        classifier = TeamClassifier().fit([red, white, red, white])
        self.assertEqual(classifier.predict(red), 0)
        self.assertEqual(classifier.predict(white), 1)

    def test_ellipse_uses_reference_arc_and_centered_id_badge(self) -> None:
        import numpy as np

        frame = np.zeros((120, 100, 3), dtype=np.uint8)
        output = draw_football_ellipse(
            frame,
            (30, 15, 50, 60),
            (0, 255, 0),
            track_id=17,
        )
        self.assertIs(output, frame)
        self.assertGreater(int(frame[55, 50, 1]), 200)
        self.assertGreater(int(frame.sum()), 0)

    def test_clean_view_uses_semantic_team_color_and_centered_id_badge(self) -> None:
        import numpy as np

        record = {
            "tracks": [
                {
                    "track_id": 12,
                    "bbox": [40, 10, 60, 70],
                    "class_name": "player",
                    "confidence": 0.91,
                    "missed": 0,
                    "age": 4,
                    "hits": 4,
                    "metadata": {
                        "team_id": "TEAM_A",
                        "team_confidence": 0.88,
                        "team_uncertain": False,
                    },
                }
            ],
            "detections": [],
        }
        clean = draw_tracking_frame(np.zeros((120, 120, 3), dtype=np.uint8), record)
        debug = draw_debug_frame(np.zeros((120, 120, 3), dtype=np.uint8), record)
        self.assertGreater(int(clean.sum()), 0)
        self.assertGreater(int(debug.sum()), 0)
        self.assertFalse((clean == debug).all())
        self.assertEqual(tuple(clean[70, 50]), TEAM_COLORS["TEAM_A"])
        edge_pixel = clean[65, 44]
        self.assertGreater(int(edge_pixel[2]), 180)
        self.assertLess(int(edge_pixel[1]), 80)
        self.assertEqual(TEAM_COLORS["TEAM_A"], (30, 40, 235))
        self.assertEqual(TEAM_COLORS["TEAM_B"], (240, 240, 240))

    def test_uncertain_team_uses_explicit_unclassified_color(self) -> None:
        import numpy as np

        record = {
            "tracks": [
                {
                    "track_id": 13,
                    "bbox": [40, 10, 60, 70],
                    "class_name": "player",
                    "confidence": 0.9,
                    "missed": 0,
                    "metadata": {
                        "team_id": "TEAM_A",
                        "team_confidence": 0.31,
                        "team_uncertain": True,
                    },
                }
            ],
            "detections": [],
        }
        clean = draw_tracking_frame(np.zeros((120, 120, 3), dtype=np.uint8), record)
        self.assertEqual(tuple(clean[70, 50]), TEAM_COLORS["UNCLASSIFIED"])

    def test_debug_status_panel_stays_clear_of_top_left_scoreboard(self) -> None:
        import numpy as np

        frame = np.zeros((400, 800, 3), dtype=np.uint8)
        debug = draw_debug_frame(
            frame,
            {
                "tracks": [],
                "detections": [],
                "camera_motion": {
                    "valid": True,
                    "method": "affine",
                    "inlier_count": 12,
                },
                "homography": {"status": "not_configured", "quality": 0.0},
            },
        )
        self.assertTrue((debug[:100, :250] == frame[:100, :250]).all())
        self.assertFalse((debug[:100, 500:] == frame[:100, 500:]).all())


class PossessionTests(unittest.TestCase):
    def test_possession_requires_repeat_and_emits_generic_team_change(self):
        from footy_cvision.core import BoundingBox, Track

        first_track = Track(
            10, BoundingBox(0, 0, 10, 20), "player", 0.9,
            metadata={"team_id": "TEAM_A", "team_uncertain": False},
        )
        second_track = Track(
            20, BoundingBox(0, 0, 10, 20), "player", 0.9,
            metadata={"team_id": "TEAM_B", "team_uncertain": False},
        )
        manager = PossessionTracker(confirm_observations=2)
        pending, events = manager.update(0, 10, 0.8, [first_track])
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(events, [])
        confirmed, events = manager.update(10, 10, 0.8, [first_track])
        self.assertEqual(confirmed["possessor_id"], 10)
        self.assertEqual(events[0]["type"], "POSSESSION_START")
        manager.update(20, 20, 0.9, [first_track, second_track])
        changed, events = manager.update(30, 20, 0.9, [first_track, second_track])
        self.assertEqual(changed["possessor_team"], "TEAM_B")
        self.assertEqual(events[0]["type"], "POSSESSION_CHANGE")

    def test_uncertain_team_label_is_not_claimed_for_possession(self) -> None:
        from footy_cvision.core import BoundingBox, Track

        for metadata in (
            {"team_id": "TEAM_A", "team_uncertain": True},
            {"team_id": "TEAM_A"},
        ):
            with self.subTest(metadata=metadata):
                track = Track(
                    42,
                    BoundingBox(0, 0, 10, 20),
                    "player",
                    0.9,
                    metadata=metadata,
                )
                manager = PossessionTracker(confirm_observations=2)
                manager.update(0, 42, 0.8, [track])
                state, events = manager.update(10, 42, 0.8, [track])
                self.assertIsNone(state["possessor_team"])
                self.assertIsNone(events[0]["team_id"])


if __name__ == "__main__":
    unittest.main()
