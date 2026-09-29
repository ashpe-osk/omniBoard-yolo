"""Football-specific classification and constraints."""

from .ball_assignment import BallAssignment, assign_ball
from .ball_trajectory import BallTrajectory, BallTrajectoryConfig
from .pitch_geometry import HomographyResult, PitchGeometry, SoccerPitchConfig
from .possession import PossessionTracker
from .team_classification import (
    TEAM_A,
    TEAM_B,
    TeamClassifier,
    TeamFeatureHistory,
    TeamStabilizer,
    extract_color_feature,
    extract_team_color_feature,
)

__all__ = [
    "BallAssignment",
    "BallTrajectory",
    "BallTrajectoryConfig",
    "HomographyResult",
    "PitchGeometry",
    "PossessionTracker",
    "SoccerPitchConfig",
    "TEAM_A",
    "TEAM_B",
    "TeamClassifier",
    "TeamFeatureHistory",
    "TeamStabilizer",
    "assign_ball",
    "extract_color_feature",
    "extract_team_color_feature",
]
