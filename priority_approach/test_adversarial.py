import pytest
import torch
import random
from constraints import RangeConstraint, MutualExclusivityConstraint, RankingConstraint

# -------------------------------
# Adversarial Attack on RangeConstraint
# -------------------------------
def adversarial_attack_range(constraint, num_products=4, batch=2, seq=3, lr=1e-2, iters=100):
    """
    Attempts to adversarially drive the output of the RangeConstraint below the lower bound.
    For each product (here we attack product 0), we maximize the violation (min_value - output)
    using gradient descent.
    """
    # Fixed constraint parameters: for instance, Setup 1.
    product_ranges = torch.tensor([[0, 1], [0.5, 2], [1, 3], [1.5, 4]])
    constraint.product_ranges = product_ranges  # override if necessary

    # Initialize an input tensor with shape (batch, seq, 2*num_products).
    # We don't care about priorities here; RangeConstraint only clamps schedules (first half).
    x = torch.rand(batch, seq, 2 * num_products, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)
    # We attack product 0.
    target_min = product_ranges[0, 0].item()

    for _ in range(iters):
        optimizer.zero_grad()
        out = constraint(x)  # output shape: (batch, seq, 2*num_products)
        schedules, _ = torch.split(out, num_products, dim=-1)
        # Our adversarial loss aims to push product 0's schedule below its min.
        # That is, we want to maximize (target_min - schedule) for product 0.
        # Equivalently, set loss = -(target_min - schedules[...,0]) = schedules[...,0] - target_min.
        loss = (schedules[..., 0] - target_min).mean()
        loss.backward()
        optimizer.step()
    # After optimization, check that product 0's output is clamped to target_min.
    final_out = constraint(x)
    final_schedules, _ = torch.split(final_out, num_products, dim=-1)
    return final_schedules[..., 0].detach()

def test_adversarial_range_constraint():
    """
    Adversarially attack RangeConstraint and verify that even after gradient descent,
    the output for a given product is not below its specified minimum.
    """
    # Create a RangeConstraint instance.
    product_ranges = torch.tensor([[0, 1], [0.5, 2], [1, 3], [1.5, 4]])
    range_constraint = RangeConstraint(number_of_products=4, product_ranges=product_ranges)
    attacked = adversarial_attack_range(range_constraint)
    # Check that attacked product 0 is exactly equal to the min.
    assert torch.allclose(attacked, torch.full_like(attacked, product_ranges[0, 0])), \
        f"Adversarial attack on RangeConstraint produced output {attacked} below min {product_ranges[0,0]}."

# -------------------------------
# Adversarial Attack on MutualExclusivityConstraint
# -------------------------------
def adversarial_attack_mutual(constraint, num_products=4, batch=2, seq=3, lr=1e-2, iters=100):
    """
    Attempts to adversarially force both products in a mutual exclusivity pair to be high.
    For a fixed pair (0, 1), we maximize the sum of both products' outputs.
    The constraint should force one to zero.
    """
    # Fix constraint parameters.
    exclusivities = torch.tensor([[0, 1]])
    constraint.exclusivities = exclusivities

    # Create an input tensor with shape (batch, seq, 2*num_products).
    # For mutual exclusivity, the module splits the input into schedules and priorities.
    x = torch.rand(batch, seq, 2 * num_products, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(iters):
        optimizer.zero_grad()
        out = constraint(x)
        schedules, _ = torch.split(out, num_products, dim=-1)
        # Our loss aims to maximize (schedules[...,0] + schedules[...,1]).
        loss = - (schedules[..., 0] + schedules[..., 1]).mean()
        loss.backward()
        optimizer.step()
    final_out = constraint(x)
    final_schedules, _ = torch.split(final_out, num_products, dim=-1)
    return final_schedules[..., 0].detach(), final_schedules[..., 1].detach()

def test_adversarial_mutual_exclusivity():
    """
    Adversarially attack MutualExclusivityConstraint for a fixed pair (0, 1)
    and verify that, even when trying to force both products high, one is forced to 0.
    """
    # Create a MutualExclusivityConstraint instance.
    exclusivities = torch.tensor([[0, 1]])
    mutual_constraint = MutualExclusivityConstraint(number_of_products=4, exclusivities=exclusivities)
    # For this module, we assume that priorities are provided as the second half of the input.
    prod0, prod1 = adversarial_attack_mutual(mutual_constraint)
    # Check that at least one of the two is forced to 0.
    # We can check that their product is 0.
    violation = (prod0 * prod1)
    assert torch.all(violation == 0), f"Adversarial attack on MutualExclusivityConstraint failed: product values {prod0} and {prod1} are both nonzero."

# -------------------------------
# Adversarial Attack on RankingConstraint
# -------------------------------
def adversarial_attack_ranking(constraint, num_products=4, batch=2, seq=3, lr=1e-2, iters=100):
    """
    Attempts to adversarially force product 0 to be higher than product 1,
    for a ranking pair (0, 1) where the constraint should enforce output[0] <= output[1].
    We try to maximize (output[0] - output[1]).
    """
    # Fix constraint parameters with a ranking pair (0, 1).
    ranking_pair = torch.tensor([[0, 1]])
    constraint.product_rankings = ranking_pair

    # Create an input tensor with shape (batch, seq, 2*num_products).
    x = torch.rand(batch, seq, 2 * num_products, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(iters):
        optimizer.zero_grad()
        out = constraint(x)
        schedules, _ = torch.split(out, num_products, dim=-1)
        # Our loss: maximize (schedules[...,0] - schedules[...,1]) i.e. we want product 0 to be greater than product 1.
        loss = - (schedules[..., 0] - schedules[..., 1]).mean()
        loss.backward()
        optimizer.step()
    final_out = constraint(x)
    final_schedules, _ = torch.split(final_out, num_products, dim=-1)
    return final_schedules[..., 0].detach(), final_schedules[..., 1].detach()

def test_adversarial_ranking():
    """
    Adversarially attack RankingConstraint for a fixed ranking pair (0, 1)
    and verify that even if we try to force product 0 above product 1,
    the module enforces product0 <= product1.
    """
    # Create a RankingConstraint instance.
    ranking_pair = torch.tensor([[0, 1]])
    ranking_constraint = RankingConstraint(number_of_products=4, product_rankings=ranking_pair)
    prod0, prod1 = adversarial_attack_ranking(ranking_constraint)
    # The ranking constraint forces product0 to be min(product0, product1),
    # so we should have prod0 <= prod1 (ideally, equality if attacked strongly).
    violation = (prod0 - prod1).clamp(min=0)
    assert torch.all(violation == 0), f"Adversarial attack on RankingConstraint failed: product0 is higher than product1 (diff {violation})."

# -------------------------------
# Run the tests if executed as script
# -------------------------------
if __name__ == "__main__":
    pytest.main(["-s"])