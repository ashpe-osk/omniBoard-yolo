"""Conservative temporally stabilized player-possession state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from footy_cvision.core import Track


@dataclass
class _PossessionSequence:
    sequence_id: int
    team_id: str | None
    start_frame: int
    end_frame: int
    players_involved: list[int] = field(default_factory=list)
    confidence_sum: float = 0.0
    observations: int = 0

    def add(
        self, frame_index: int, player_id: int, confidence: float
    ) -> None:
        self.end_frame = frame_index
        if not self.players_involved or self.players_involved[-1] != player_id:
            self.players_involved.append(player_id)
        self.confidence_sum += confidence
        self.observations += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "team_id": self.team_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_source_frames": self.end_frame - self.start_frame,
            "players_involved": list(self.players_involved),
            "mean_confidence": (
                self.confidence_sum / self.observations
                if self.observations
                else 0.0
            ),
        }


class PossessionTracker:
    """Requires repeated proximity evidence before confirming control."""

    def __init__(
        self,
        confirm_observations: int = 2,
        release_observations: int = 2,
        minimum_confidence: float = 0.10,
    ) -> None:
        if confirm_observations < 1 or release_observations < 1:
            raise ValueError("Possession observation counts must be positive")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        self.confirm_observations = confirm_observations
        self.release_observations = release_observations
        self.minimum_confidence = minimum_confidence
        self._current_player: int | None = None
        self._current_team: str | None = None
        self._pending_player: int | None = None
        self._pending_count = 0
        self._pending_confidence = 0.0
        self._unobserved_count = 0
        self._sequence_id = 0
        self._sequence: _PossessionSequence | None = None
        self.completed_sequences: list[dict[str, Any]] = []

    def update(
        self,
        frame_index: int,
        candidate_player_id: int | None,
        candidate_confidence: float,
        tracks: list[Track],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        active = {
            track.track_id: track
            for track in tracks
            if track.missed == 0
        }
        candidate = (
            active.get(candidate_player_id)
            if candidate_player_id is not None
            else None
        )
        valid_candidate = (
            candidate is not None
            and candidate_confidence >= self.minimum_confidence
        )
        events: list[dict[str, Any]] = []

        if valid_candidate and candidate is not None:
            candidate_id = candidate.track_id
            self._unobserved_count = 0
            if candidate_id == self._current_player:
                self._pending_player = None
                self._pending_count = 0
                self._pending_confidence = 0.0
                self._record_current(frame_index, candidate_id, candidate_confidence)
            else:
                if self._pending_player == candidate_id:
                    self._pending_count += 1
                    self._pending_confidence += candidate_confidence
                else:
                    self._pending_player = candidate_id
                    self._pending_count = 1
                    self._pending_confidence = candidate_confidence
                if self._pending_count >= self.confirm_observations:
                    confirmed_confidence = (
                        self._pending_confidence / self._pending_count
                    )
                    events.extend(
                        self._confirm(
                            frame_index,
                            candidate,
                            confirmed_confidence,
                        )
                    )
        else:
            self._pending_player = None
            self._pending_count = 0
            self._pending_confidence = 0.0
            self._unobserved_count += 1
            if self._unobserved_count >= self.release_observations:
                self._finish_sequence()
                self._current_player = None
                self._current_team = None

        sequence = self._sequence.to_dict() if self._sequence is not None else None
        state = {
            "frame_index": frame_index,
            "possessor_id": self._current_player,
            "possessor_team": self._current_team,
            "confidence": (
                self._sequence.to_dict()["mean_confidence"]
                if self._sequence is not None
                else 0.0
            ),
            "status": (
                "confirmed"
                if self._current_player is not None and self._unobserved_count == 0
                else "held"
                if self._current_player is not None
                else "pending"
                if self._pending_player is not None
                else "unobserved"
            ),
            "candidate_id": self._pending_player,
            "candidate_observations": self._pending_count,
            "sequence": sequence,
        }
        return state, events

    def _confirm(
        self, frame_index: int, track: Track, confidence: float
    ) -> list[dict[str, Any]]:
        player_id = track.track_id
        team_id = (
            track.metadata.get("team_id")
            if track.metadata.get("team_uncertain") is False
            else None
        )
        previous_player = self._current_player
        previous_team = self._current_team
        events: list[dict[str, Any]] = []
        if previous_player is None:
            event_type = "POSSESSION_START"
        elif (
            team_id is None
            or previous_team is None
            or team_id != previous_team
        ):
            event_type = "POSSESSION_CHANGE"
            self._finish_sequence()
        else:
            event_type = "PLAYER_CONTROL_CHANGE"
        if self._sequence is None:
            self._sequence_id += 1
            self._sequence = _PossessionSequence(
                sequence_id=self._sequence_id,
                team_id=team_id,
                start_frame=frame_index,
                end_frame=frame_index,
            )
        self._sequence.add(frame_index, player_id, confidence)
        self._current_player = player_id
        self._current_team = team_id
        events.append(
            {
                "type": event_type,
                "frame_index": frame_index,
                "previous_player_id": previous_player,
                "player_id": player_id,
                "team_id": team_id,
                "confidence": confidence,
                "ground_truth": False,
            }
        )
        self._pending_player = None
        self._pending_count = 0
        self._pending_confidence = 0.0
        return events

    def _record_current(
        self, frame_index: int, player_id: int, confidence: float
    ) -> None:
        if self._sequence is not None:
            self._sequence.add(frame_index, player_id, confidence)

    def _finish_sequence(self) -> None:
        if self._sequence is not None:
            self.completed_sequences.append(self._sequence.to_dict())
            self._sequence = None

    def report(self) -> dict[str, Any]:
        sequences = [*self.completed_sequences]
        if self._sequence is not None:
            sequences.append(self._sequence.to_dict())
        return {
            "sequences": sequences,
            "sequence_count": len(sequences),
            "possession_changes": sum(
                sequence["team_id"] != prior["team_id"]
                for prior, sequence in zip(sequences, sequences[1:])
            ),
            "pass_count": None,
            "carry_count": None,
            "shot_count": None,
            "unsupported_events": {
                "passes": "No confirmed ball-flight and receiver evidence is implemented.",
                "carries": "No ball-control movement model is implemented.",
                "shots": "Pitch/goal geometry and reliable ball trajectories are unavailable.",
            },
        }
