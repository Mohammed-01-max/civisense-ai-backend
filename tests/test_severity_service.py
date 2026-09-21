import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5432/civicsafe",
)
os.environ.setdefault("SECRET_KEY", "N7v!2xQ9#Lm4Rk8pW3zD6hJ1cF5yT0uA")
os.environ.setdefault(
    "CORS_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080",
)

import ai.severity as severity


class SeverityServiceTests(unittest.TestCase):
    def setUp(self):
        self.original_service = severity._service
        severity._service = None

    def tearDown(self):
        severity._service = self.original_service

    def test_none_input_does_not_load_service(self):
        with patch.object(severity, "get_severity_service") as get_service:
            self.assertIsNone(severity.predict_severity(None))
        get_service.assert_not_called()

    def test_blank_input_does_not_load_service(self):
        with patch.object(severity, "get_severity_service") as get_service:
            self.assertIsNone(severity.predict_severity("  \n\t "))
        get_service.assert_not_called()

    def test_valid_prediction_uses_max_probability_and_source(self):
        service = severity.SeverityService.__new__(severity.SeverityService)
        service._tfidf = Mock()
        service._tfidf.transform.return_value = object()
        service._booster = Mock()
        service._booster.predict.return_value = np.array([[0.1, 0.7, 0.2]])

        with patch("xgboost.DMatrix", return_value=object()):
            result = service.predict("  broken streetlight  ")

        self.assertEqual(result.severity, "MEDIUM")
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.source, "xgboost-tfidf")
        service._tfidf.transform.assert_called_once_with(["broken streetlight"])

    def test_all_class_ids_map_to_expected_labels(self):
        for class_id, expected_label in severity._LABEL_MAP.items():
            service = severity.SeverityService.__new__(severity.SeverityService)
            service._tfidf = Mock()
            service._tfidf.transform.return_value = object()
            service._booster = Mock()
            probabilities = np.array([[0.1, 0.2, 0.3]])
            probabilities[0, class_id] = 0.9
            service._booster.predict.return_value = probabilities

            with self.subTest(class_id=class_id), patch(
                "xgboost.DMatrix", return_value=object()
            ):
                result = service.predict("valid complaint")

            self.assertEqual(result.severity, expected_label)
            self.assertEqual(result.confidence, 0.9)

    def test_invalid_probability_shape_is_rejected(self):
        service = severity.SeverityService.__new__(severity.SeverityService)
        service._tfidf = Mock()
        service._tfidf.transform.return_value = object()
        service._booster = Mock()
        service._booster.predict.return_value = np.array([0.1, 0.9, 0.0])

        with patch("xgboost.DMatrix", return_value=object()), self.assertRaises(
            RuntimeError
        ):
            service.predict("valid complaint")

    def test_nonfinite_probabilities_are_rejected(self):
        service = severity.SeverityService.__new__(severity.SeverityService)
        service._tfidf = Mock()
        service._tfidf.transform.return_value = object()
        service._booster = Mock()
        service._booster.predict.return_value = np.array([[np.nan, 0.5, 0.5]])

        with patch("xgboost.DMatrix", return_value=object()), self.assertRaises(
            RuntimeError
        ):
            service.predict("valid complaint")

    def test_artifacts_load_lazily_and_only_once(self):
        fake_service = Mock()
        with patch.object(severity, "SeverityService", return_value=fake_service) as constructor:
            with patch.object(severity.settings, "SEVERITY_ARTIFACT_DIR", "C:/frozen"):
                first = severity.get_severity_service()
                second = severity.get_severity_service()

        self.assertIs(first, fake_service)
        self.assertIs(second, fake_service)
        constructor.assert_called_once_with(Path("C:/frozen"))

    def test_artifact_load_failure_returns_none(self):
        with patch.object(
            severity, "SeverityService", side_effect=FileNotFoundError("missing")
        ):
            with patch.object(severity.settings, "SEVERITY_ARTIFACT_DIR", "C:/missing"):
                self.assertIsNone(severity.get_severity_service())

    def test_inference_failure_returns_none(self):
        failing_service = Mock()
        failing_service.predict.side_effect = RuntimeError("inference failed")
        with patch.object(severity, "get_severity_service", return_value=failing_service):
            self.assertIsNone(severity.predict_severity("valid complaint"))

    def test_frozen_real_artifact_inference(self):
        artifact_dir = Path(__file__).resolve().parents[1] / "ai" / "severity_artifact"
        service = severity.SeverityService(artifact_dir)
        result = service.predict("The road has dangerous potholes and needs urgent repair")

        self.assertIn(result.severity, {"LOW", "MEDIUM", "HIGH"})
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.source, "xgboost-tfidf")


if __name__ == "__main__":
    unittest.main()