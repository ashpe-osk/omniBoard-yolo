import unittest

from footy_cvision.core import BoundingBox, Detection
from footy_cvision.detection.filter import DetectionFilter
from footy_cvision.evaluation.metrics import detection_metrics
from footy_cvision.football.team_classification import (
    TEAM_A,
    TEAM_B,
    TeamClassifier,
    TeamFeatureHistory,
    TeamStabilizer,
)
from footy_cvision.tracking.association import AssociationConfig
from footy_cvision.tracking.tracker import Tracker, TrackerConfig


class CoreTests(unittest.TestCase):
    def test_box_iou_and_invalid_confidence(self):
        first = BoundingBox(0, 0, 10, 10)
        self.assertEqual(first.iou(BoundingBox(0, 0, 10, 10)), 1.0)
        with self.assertRaises(ValueError):
            Detection(first, 1.5, "player")

    def test_filter_confidence_class_and_pitch(self):
        detections = [
            Detection(BoundingBox(1, 1, 4, 8), 0.9, "player"),
            Detection(BoundingBox(20, 20, 24, 28), 0.9, "player"),
            Detection(BoundingBox(1, 1, 4, 8), 0.1, "player"),
        ]
        selected = DetectionFilter(
            min_confidence=0.5,
            allowed_classes={"player"},
            pitch_polygon=((0, 0), (10, 0), (10, 10), (0, 10)),
            filter_out_of_bounds=True,
        )(detections)
        self.assertEqual(len(selected), 1)

    def test_filter_removes_nested_duplicate_detection(self):
        detections = [
            Detection(BoundingBox(100, 100, 220, 420), 0.8, "player"),
            Detection(BoundingBox(104, 104, 196, 282), 0.3, "player"),
            Detection(BoundingBox(235, 100, 355, 420), 0.8, "player"),
        ]
        selected = DetectionFilter(
            min_confidence=0.1,
            nms_iou_threshold=0.5,
            containment_threshold=0.9,
        )(detections)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].confidence, 0.8)

    def test_filter_suppresses_similar_person_boxes_with_aligned_footpoints(self):
        detections = [
            Detection(
                BoundingBox(263.186, 784.576, 299.944, 912.649),
                0.41,
                "player",
            ),
            Detection(
                BoundingBox(222.566, 708.143, 296.325, 892.344),
                0.38,
                "player",
            ),
        ]
        selected = DetectionFilter(
            nms_iou_threshold=0.5,
            containment_threshold=0.9,
            person_duplicate_containment_threshold=0.75,
            person_duplicate_footpoint_distance_ratio=0.2,
        )(detections)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].confidence, 0.41)

    def test_person_containment_does_not_suppress_distant_footpoints(self):
        detections = [
            Detection(BoundingBox(0, 0, 100, 300), 0.9, "player"),
            Detection(BoundingBox(10, 0, 90, 200), 0.8, "player"),
        ]
        selected = DetectionFilter(
            nms_iou_threshold=None,
            containment_threshold=None,
            person_duplicate_containment_threshold=0.75,
            person_duplicate_footpoint_distance_ratio=0.2,
        )(detections)
        self.assertEqual(len(selected), 2)

    def test_person_duplicate_suppression_respects_appearance_mismatch(self):
        detections = [
            Detection(
                BoundingBox(100, 100, 220, 420),
                0.9,
                "player",
                appearance=(1.0, 0.0),
            ),
            Detection(
                BoundingBox(104, 104, 196, 282),
                0.8,
                "player",
                appearance=(0.0, 1.0),
            ),
        ]
        selected = DetectionFilter(
            nms_iou_threshold=None,
            containment_threshold=None,
            person_duplicate_containment_threshold=0.75,
            person_duplicate_footpoint_distance_ratio=0.2,
            person_duplicate_center_distance_ratio=0.5,
            person_duplicate_min_height_ratio=0.55,
            person_duplicate_max_appearance_distance=0.15,
        )(detections)
        self.assertEqual(len(selected), 2)

    def test_tracker_keeps_id_over_short_occlusion(self):
        tracker = Tracker(TrackerConfig(max_age=2))
        first = tracker.update(
            [Detection(BoundingBox(0, 0, 10, 20), 0.9, "player")], 0
        )[0]
        self.assertEqual(tracker.update([], 1), [])
        recovered = tracker.update(
            [Detection(BoundingBox(2, 0, 12, 20), 0.8, "player")], 2
        )[0]
        self.assertEqual(first.track_id, recovered.track_id)
        self.assertEqual(recovered.hits, 2)

    def test_team_classifier_is_deterministic(self):
        classifier = TeamClassifier().fit(
            [(0, 0, 0), (1, 1, 1), (250, 0, 0), (255, 2, 1)]
        )
        self.assertEqual(classifier.predict((2, 1, 0)), 0)
        self.assertEqual(classifier.predict((250, 1, 0)), 1)

    def test_team_classifier_maps_kit_features_to_stable_semantic_ids(self):
        red = [0.0] * 144
        red[7] = 1.0
        white = [0.0] * 144
        white[0] = 1.0
        classifier = TeamClassifier().fit([white, red, white, red])
        reverse_order = TeamClassifier().fit([red, white, red, white])
        red_team, confidence = classifier.predict_with_confidence(red)
        white_team, _ = classifier.predict_with_confidence(white)
        self.assertEqual(red_team, 0)
        self.assertEqual(white_team, 1)
        self.assertEqual(
            red_team,
            reverse_order.predict_with_confidence(red)[0],
        )
        self.assertGreater(confidence, 0.9)

    def test_team_stabilizer_keeps_track_assignment_until_repeated_evidence(self):
        stabilizer = TeamStabilizer()
        first = stabilizer.update(54, TEAM_A, 0.8)
        self.assertEqual(first["team_id"], TEAM_A)
        self.assertTrue(first["team_uncertain"])
        stabilizer.update(54, TEAM_B, 0.9)
        second = stabilizer.update(54, TEAM_B, 0.9)
        self.assertEqual(second["team_id"], TEAM_A)
        switched = stabilizer.update(54, TEAM_B, 0.9)
        self.assertEqual(switched["team_id"], TEAM_B)
        self.assertEqual(switched["team_history"], [TEAM_A, TEAM_B])
        self.assertFalse(switched["team_uncertain"])

    def test_team_feature_history_smooths_single_frame_color_noise(self):
        history = TeamFeatureHistory(window_size=5)
        history.update(7, (1.0, 0.0))
        history.update(7, (1.0, 0.0))
        smoothed = history.update(7, (0.0, 1.0))
        self.assertAlmostEqual(smoothed[0], 2 / 3)
        self.assertAlmostEqual(smoothed[1], 1 / 3)
        with self.assertRaises(ValueError):
            history.update(7, (1.0, 0.0, 0.0))

    def test_team_stabilizer_adapts_to_a_track_identity_change(self):
        stabilizer = TeamStabilizer()
        for _ in range(5):
            stabilizer.update(33, TEAM_A, 0.9)
        result = None
        for _ in range(6):
            result = stabilizer.update(33, TEAM_B, 0.9)
        self.assertIsNotNone(result)
        self.assertEqual(result["team_id"], TEAM_B)
        self.assertEqual(result["team_history"], [TEAM_A, TEAM_B])

    def test_team_stabilizer_marks_competing_recent_evidence_uncertain(self):
        stabilizer = TeamStabilizer()
        for _ in range(4):
            result = stabilizer.update(33, TEAM_A, 0.9)
        self.assertFalse(result["team_uncertain"])
        result = stabilizer.update(33, TEAM_B, 0.95)
        self.assertFalse(result["team_uncertain"])
        result = stabilizer.update(33, TEAM_B, 0.95)
        self.assertEqual(result["team_id"], TEAM_A)
        self.assertTrue(result["team_uncertain"])

    def test_team_feature_history_can_reset_after_a_confident_appearance_change(self):
        history = TeamFeatureHistory(window_size=10)
        for _ in range(8):
            history.update(33, (0.1, 0.1, 0.8))
        changed = history.reset(33, (0.8, 0.8, 0.8))
        self.assertEqual(changed, (0.8, 0.8, 0.8))
        self.assertEqual(
            history.update(33, (0.8, 0.8, 0.8)),
            (0.8, 0.8, 0.8),
        )

    def test_camera_motion_compensates_tracker_association(self):
        tracker = Tracker(
            TrackerConfig(
                max_age=2,
                association=AssociationConfig(
                    max_center_distance=20,
                    appearance_weight=0.0,
                ),
            )
        )
        first = tracker.update(
            [Detection(BoundingBox(10, 10, 20, 40), 0.9, "player")],
            0,
        )[0]
        moved = tracker.update(
            [Detection(BoundingBox(60, 10, 70, 40), 0.9, "player")],
            1,
            camera_transform=((1.0, 0.0, 50.0), (0.0, 1.0, 0.0)),
        )[0]
        self.assertEqual(first.track_id, moved.track_id)
        self.assertEqual(moved.bbox, BoundingBox(60, 10, 70, 40))
        self.assertEqual(tracker.last_camera_compensated_track_count, 1)

    def test_detection_precision_and_recall(self):
        box = BoundingBox(0, 0, 10, 10)
        actual = Detection(box, 1.0, "player")
        result = detection_metrics([[actual]], [[actual]])
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
