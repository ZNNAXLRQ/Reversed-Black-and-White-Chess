"""NCS costs from fresh terminal play on one immutable recorded-position panel."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from optimization.objective import EvaluationResult

from .continuation_dataset import verify_training
from .match_runner import MatchRunner, WeightedAgent
from .reversi_objective import ReversiObjective


class ContinuationObjective(ReversiObjective):
    """Shares weight decoding only; does not initialize the legacy match objective."""

    def __init__(self, protocol, audit_path=None):
        verify_training(protocol)
        # Deep, private snapshot: caller edits cannot invalidate cached parent costs.
        self._protocol_json = json.dumps(protocol, allow_nan=False)
        self.runner = MatchRunner(depth=protocol["depth"], node_limit=protocol["node_limit"])
        self.audit_path = Path(audit_path) if audit_path is not None else None
        if self.audit_path is not None:
            with self.audit_path.open("x", encoding="utf-8"):
                pass
        self._cache = {}
        self.games_played = 0
        self.cache_hits = 0

    def __call__(self, candidate):
        weights = self.decode(candidate)
        protocol = json.loads(self._protocol_json)
        if (self.runner.depth != protocol["depth"]
                or self.runner.node_limit != protocol["node_limit"]):
            raise ValueError("search budget changed within a frozen optimization")
        key = tuple(weights)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        per_opponent, per_stage, per_game = [], {}, []
        evaluation = len(self._cache)
        stream = self.audit_path.open("a", encoding="utf-8") if self.audit_path else None
        try:
            for opponent in protocol["opponents"]:
                utilities = []
                for index, position in enumerate(protocol["positions"]):
                    for color in (-1, 1):
                        agent = WeightedAgent(weights.copy(), self.runner.depth, self.runner.node_limit)
                        other = WeightedAgent(opponent["weights"], self.runner.depth, self.runner.node_limit)
                        black, white = (agent, other) if color == -1 else (other, agent)
                        game = self.runner.play_game(black, white, opening=position)
                        self.games_played += 1
                        u = (game.white_count - game.black_count) / 64.0 * (-color)
                        utilities.append(u)
                        per_stage.setdefault((opponent["id"], position["stage"]), []).append(u)
                        metadata = dict(position_index=index, game_id=position["game_id"],
                                        ply=position["ply"], canonical_sha256=position["canonical_sha256"],
                                        side_to_move=position["color"], candidate_color=color,
                                        opponent_id=opponent["id"], stage=position["stage"], u=u)
                        per_game.append(dict(metadata, black_count=game.black_count,
                                             white_count=game.white_count, steps=game.steps))
                        if stream is not None:
                            stream.write(json.dumps(dict(type="game", evaluation=evaluation,
                                                         weights=weights.tolist(), **metadata,
                                                         result=asdict(game)), allow_nan=False) + "\n")
                            stream.flush()
                values = np.asarray(utilities)
                per_opponent.append(dict(id=opponent["id"], weight=opponent["weight"],
                                         mean_u=float(values.mean()), games=len(values),
                                         wins=int(np.count_nonzero(values > 0)),
                                         draws=int(np.count_nonzero(values == 0)),
                                         losses=int(np.count_nonzero(values < 0))))
            mean_u = float(sum(row["weight"] * row["mean_u"] for row in per_opponent))
            metrics = dict(weights=weights.tolist(), protocol_sha256=protocol["sha256"],
                           mean_u=mean_u, incumbent_mean_u=per_opponent[0]["mean_u"],
                           games=len(per_game), per_opponent=per_opponent, per_game=per_game,
                           per_stage=[dict(opponent_id=opp, stage=stage, mean_u=float(np.mean(us)))
                                      for (opp, stage), us in per_stage.items()])
            result = EvaluationResult(cost=(1 - mean_u) / 2, metrics=metrics)
            if stream is not None:
                stream.write(json.dumps(dict(type="completed", evaluation=evaluation,
                                             cost=result.cost, metrics=metrics), allow_nan=False) + "\n")
                stream.flush()
        finally:
            if stream is not None:
                stream.close()
        self._cache[key] = result
        return result

    def cached_result(self, candidate):
        """Reporting must not trigger additional training games."""
        return self._cache[tuple(self.decode(candidate))]