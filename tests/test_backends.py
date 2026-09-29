import unittest

from footy_cvision.models.backends import normalize_detections


class BackendNormalizationTests(unittest.TestCase):
    def test_roboflow_prediction_mapping(self):
        detections = normalize_detections(
            {
                "predictions": [
                    {
                        "x": 20,
                        "y": 30,
                        "width": 10,
                        "height": 12,
                        "confidence": 0.8,
                        "class": "player",
                    }
                ]
            }
        )
        self.assertEqual(detections[0].bbox.to_list(), [15.0, 24.0, 25.0, 36.0])
        self.assertEqual(detections[0].class_name, "player")

    def test_ultralytics_mapping_with_class_index(self):
        detections = normalize_detections(
            [
                {
                    "xyxy": [1, 2, 11, 22],
                    "conf": 0.95,
                    "cls": 0,
                }
            ],
            class_names={0: "ball"},
        )
        self.assertEqual(detections[0].bbox.to_list(), [1.0, 2.0, 11.0, 22.0])
        self.assertEqual(detections[0].class_name, "ball")


if __name__ == "__main__":
    unittest.main()
