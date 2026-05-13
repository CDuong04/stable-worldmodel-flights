"""Smoke test for stable_worldmodel.clf_cost on synthetic data.

Validates the CLF-augmented cost wrapper end-to-end without needing a
trained world-model checkpoint. Runs in seconds on CPU.

Checks:
  1. lambda_V=0 reproduces the base cost byte-for-byte
  2. lambda_V>0 adds non-negative penalty
  3. The penalty fires when V increases along the rollout, doesn't fire when V decreases
  4. make_rocket_V matches LyapunovMonitor.compute_V on identical inputs
  5. Output shape matches the base cost shape (CEM contract)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stable_worldmodel.clf_cost import (
    CLFAugmentedCost,
    CLFCostConfig,
    make_rocket_V,
)
from stable_worldmodel.lyapunov import LyapunovMonitor, LyapunovConfig


# ---------------------------------------------------------------------------
# Fakes that mimic dinowm's cost-model contract well enough for the wrapper.
# ---------------------------------------------------------------------------
class _FakeBaseCost:
    """Mimics dinowm.get_cost: writes 'predicted_pixels_embed' into info_dict
    during the call and returns a (B,) cost tensor."""

    def __init__(self, B, T, P, d):
        self.B, self.T, self.P, self.d = B, T, P, d

    def get_cost(self, info_dict, action_candidates):
        # Pretend we ran encode + rollout: stash a synthetic (B, T, P, d) tensor
        info_dict["predicted_pixels_embed"] = info_dict["__test_predicted__"]
        return info_dict["__test_base_cost__"]

    def eval(self):
        return self

    def to(self, device):
        return self


class _IdentityProbe(nn.Module):
    """state_dim must be >= 13 to satisfy the rocket V index layout."""

    def __init__(self, embed_dim=32, state_dim=17):
        super().__init__()
        self.linear = nn.Linear(embed_dim, state_dim)
        # Initialise so the decoded x_hat is a deterministic linear function of z
        torch.manual_seed(0)
        nn.init.kaiming_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z):
        return self.linear(z)


def _check(name, fn):
    try:
        fn()
        print(f"  [OK]  {name}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return False


def main():
    failures = 0
    print("== 1. lambda_V=0 reproduces base cost exactly ==")

    def lambda_zero():
        torch.manual_seed(1)
        B, T, P, d = 4, 6, 8, 32
        base = _FakeBaseCost(B, T, P, d)
        probe = _IdentityProbe(embed_dim=d, state_dim=17)
        wrap = CLFAugmentedCost(base_cost_model=base, decoder=probe,
                                compute_V=make_rocket_V(),
                                cfg=CLFCostConfig(lambda_V=0.0))
        info = {
            "__test_predicted__": torch.randn(B, T, P, d),
            "__test_base_cost__": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
        out = wrap.get_cost(info, action_candidates=torch.randn(B, T, 7))
        assert torch.allclose(out, info["__test_base_cost__"]), (out, info["__test_base_cost__"])

    failures += not _check("lambda_V=0 short-circuits to base", lambda_zero)

    print("\n== 2. lambda_V>0 adds non-negative penalty ==")

    def lambda_positive_nonneg():
        torch.manual_seed(2)
        B, T, P, d = 4, 6, 8, 32
        base = _FakeBaseCost(B, T, P, d)
        probe = _IdentityProbe(embed_dim=d, state_dim=17)
        wrap = CLFAugmentedCost(base_cost_model=base, decoder=probe,
                                compute_V=make_rocket_V(),
                                cfg=CLFCostConfig(lambda_V=1.0, eta=0.0))
        info = {
            "__test_predicted__": torch.randn(B, T, P, d),
            "__test_base_cost__": torch.zeros(B),
        }
        out = wrap.get_cost(info, action_candidates=torch.randn(B, T, 7))
        # base cost is zero -> any non-zero output is the penalty
        assert (out >= 0).all(), f"penalty went negative: {out}"
        # At random latents we expect at least one candidate to have an
        # ascending V at some step -> penalty > 0 in aggregate
        assert out.sum() > 0, f"penalty was always zero: {out}"
        print(f"    penalty range = [{out.min().item():.4f}, {out.max().item():.4f}]")

    failures += not _check("lambda_V>0 penalty is non-negative and non-trivial",
                          lambda_positive_nonneg)

    print("\n== 3. Penalty fires on ascending V, vanishes on descending V ==")

    def descent_vs_ascent():
        torch.manual_seed(3)
        B, T, P, d = 1, 4, 1, 17
        # Build TWO synthetic latent trajectories:
        #   trajectory A: x_hat sequence such that pos -> 0 monotonically ->
        #                  V monotonically descending -> penalty == 0
        #   trajectory B: x_hat sequence with pos away from pad ->
        #                  V monotonically ascending  -> penalty > 0
        # Use the identity probe (z = x_hat for the position dims).
        # state_dim 17 with the standard rocket index layout from LyapunovConfig.
        # We pack the x_hat we want into z directly via a hand-crafted probe.
        class Direct(nn.Module):
            def forward(self, z):
                return z  # z IS x_hat
        probe = Direct()

        # Construct V_traj_A descending: pos goes 4,3,2,1 in x.
        z_A = torch.zeros(1, T, 1, 17)
        for k in range(T):
            z_A[0, k, 0, 0] = 4.0 - k    # pos_x
        # Construct V_traj_B ascending: pos goes 1,2,3,4 in x.
        z_B = torch.zeros(1, T, 1, 17)
        for k in range(T):
            z_B[0, k, 0, 0] = 1.0 + k

        base = _FakeBaseCost(1, T, 1, 17)
        wrap = CLFAugmentedCost(base_cost_model=base, decoder=probe,
                                compute_V=make_rocket_V(),
                                cfg=CLFCostConfig(lambda_V=1.0, eta=0.0,
                                                  normalized_descent=False))
        info_A = {"__test_predicted__": z_A, "__test_base_cost__": torch.zeros(1)}
        info_B = {"__test_predicted__": z_B, "__test_base_cost__": torch.zeros(1)}
        out_A = wrap.get_cost(info_A, action_candidates=torch.randn(1, T, 7))
        out_B = wrap.get_cost(info_B, action_candidates=torch.randn(1, T, 7))
        assert out_A.item() == 0.0, f"descending V should have zero penalty, got {out_A}"
        assert out_B.item() > 0, f"ascending V should have positive penalty, got {out_B}"
        # And specifically: each step k has V[k+1] - V[k] = (k+2)^2 - (k+1)^2 = 2k+3
        # sum over k=0..T-2 = 3 + 5 + 7 = 15 (for T=4)
        assert abs(out_B.item() - 15.0) < 1e-5, f"expected exact sum 15, got {out_B.item()}"
        print(f"    descending V -> penalty {out_A.item():.4f} (expected 0)")
        print(f"    ascending  V -> penalty {out_B.item():.4f} (expected 15.0)")

    failures += not _check("descent ok / ascent triggers penalty (exact value)",
                          descent_vs_ascent)

    print("\n== 4. make_rocket_V matches LyapunovMonitor.compute_V ==")

    def V_matches_monitor():
        torch.manual_seed(4)
        x = torch.randn(8, 17)
        V1 = make_rocket_V(alpha=0.5, beta=0.1, pad_pos=(0., 0., 0.))(x)
        # Build a LyapunovMonitor solely to call compute_V (it doesn't need real ckpt)
        class _Identity(nn.Module):
            def forward(self, z):
                return z
        mon = LyapunovMonitor(decoder=_Identity(), cfg=LyapunovConfig(
            alpha=0.5, beta=0.1, pad_pos=(0., 0., 0.)), device="cpu")
        V2 = mon.compute_V(x)
        assert torch.allclose(V1, V2, atol=1e-6), (V1, V2)
        print(f"    max|V1 - V2| = {(V1 - V2).abs().max().item():.2e}")

    failures += not _check("rocket V identical to LyapunovMonitor.compute_V",
                          V_matches_monitor)

    print("\n== 5. Output shape matches base cost shape ==")

    def shape_preserved():
        torch.manual_seed(5)
        B, T, P, d = 5, 4, 8, 32
        base = _FakeBaseCost(B, T, P, d)
        probe = _IdentityProbe(embed_dim=d, state_dim=17)
        wrap = CLFAugmentedCost(base_cost_model=base, decoder=probe,
                                compute_V=make_rocket_V(),
                                cfg=CLFCostConfig(lambda_V=0.5))
        for base_shape in [(B,), (1, B), (B, 1)]:
            info = {
                "__test_predicted__": torch.randn(B, T, P, d),
                "__test_base_cost__": torch.randn(*base_shape),
            }
            out = wrap.get_cost(info, action_candidates=torch.randn(B, T, 7))
            assert out.shape == base_shape, f"shape mismatch base={base_shape} out={out.shape}"
            print(f"    base.shape={base_shape} -> out.shape={tuple(out.shape)} (OK)")

    failures += not _check("output shape == base shape across (B,) (1,B) (B,1)",
                          shape_preserved)

    print("\n== 6. diagnose_plan reports predicted CLF traces ==")

    def diagnose_plan():
        torch.manual_seed(6)
        B, T, P, d = 2, 5, 3, 17

        class Direct(nn.Module):
            def forward(self, z):
                return z

        pred = torch.zeros(B, T, P, d)
        pred[0, :, :, 0] = torch.arange(T, 0, -1).view(T, 1)
        pred[1, :, :, 0] = torch.arange(1, T + 1).view(T, 1)

        base = _FakeBaseCost(B, T, P, d)
        wrap = CLFAugmentedCost(base_cost_model=base, decoder=Direct(),
                                compute_V=make_rocket_V(),
                                cfg=CLFCostConfig(lambda_V=2.0, eta=0.0))
        info = {
            "__test_predicted__": pred,
            "__test_base_cost__": torch.tensor([1.0, 2.0]),
        }
        diag = wrap.diagnose_plan(info, action_candidates=torch.randn(B, T, 7))
        assert diag["available"] is True
        assert len(diag["pred_V_trace"]) == B
        assert len(diag["pred_delta_trace"][0]) == T - 1
        assert diag["pred_violation_rate"][0] == 0.0
        assert diag["pred_violation_rate"][1] > 0.0
        assert diag["clf_penalty"][1] > diag["clf_penalty"][0]
        print(f"    pred violation rates = {diag['pred_violation_rate']}")

    failures += not _check("diagnose_plan emits predicted V/delta summaries",
                          diagnose_plan)

    print()
    if failures == 0:
        print("All CLF-cost smoke checks PASSED.")
        return 0
    print(f"{failures} CLF-cost smoke checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
