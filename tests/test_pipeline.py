import json
import unittest

from footy_cvision.pipeline.pipeline import (
    Pipeline,
    PipelineConfig,
    records_to_jsonable,
)


class PipelineTests(unittest.TestCase):
    def test_pipeline_accepts_detector_callback_and_serializes(self):
        pipeline = Pipeline(detector=lambda frame: frame["detections"])
        results = pipeline.run(
            [
                {
                    "detections": [
                        {
                            "bbox": [10, 10, 20, 30],
                            "confidence": 0.9,
                            "class_name": "player",
                        },
                        {
                            "bbox": [14, 25, 16, 27],
                            "confidence": 0.8,
                            "class_name": "ball",
                        },
                    ]
                }
            ]
        )
        records = records_to_jsonable(results)
        encoded = json.dumps(records)
        self.assertIn('"frame_index": 0', encoded)
        self.assertEqual(records[0]["ball_assignment"]["player_id"], 1)
        self.assertEqual(len(records[0]["tracks"]), 2)

    def test_pipeline_without_detector_requires_inline_detections(self):
        with self.assertRaises(ValueError):
            Pipeline().process_frame({}, 0)

    def test_outside_pitch_ball_is_filtered_and_reported(self):
        pipeline = Pipeline(
            config=PipelineConfig(
                pitch_polygon=((0, 0), (100, 0), (100, 100), (0, 100))
            )
        )
        result = pipeline.process_frame(
            {
                "detections": [
                    {
                        "bbox": [150, 150, 155, 155],
                        "confidence": 0.9,
                        "class_name": "ball",
                    }
                ]
            },
            0,
        )
        self.assertEqual(result.detections, [])
        self.assertIn("ball:outside_pitch", result.violations)


if __name__ == "__main__":
    unittest.main()
