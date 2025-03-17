import pytest
import torch
import random
from constraints import ConstraintModule

def check_range_with_exceptions(output, product_ranges, exclusivities, rankings):
    """
    For each product in the output (shape: [batch, seq, num_products]), verify that:
      - If the value is within [min, max] for that product, it is valid.
      - If the value is above max, it is invalid.
      - If the value is below the nominal min, then it is allowed only if one of the following holds:
           (a) Mutual exclusivity forces it to 0 (i.e. the value is exactly 0),
           (b) A ranking rule forces it below its nominal min—that is, if product i is the first element in 
               a ranking pair (i, j) and the value of product j is below product i’s min.
    Returns a tuple (True, "") if all values pass; otherwise (False, <error message>).
    """
    batch, seq, num_products = output.shape
    for b in range(batch):
        for t in range(seq):
            for i in range(num_products):
                val = output[b, t, i].item()
                min_val = product_ranges[i, 0].item()
                max_val = product_ranges[i, 1].item()
                # If within range, it's valid.
                if min_val <= val <= max_val:
                    continue
                # If above max, that's an error.
                if val > max_val:
                    return False, f"Batch {b}, time {t}, product {i}: {val} exceeds max {max_val}."
                # If below nominal min, check exceptions.
                if val < min_val:
                    ranking_forced = False
                    # Check ranking: if i is the first element in a ranking pair (i,j) and product j's value is below i's min.
                    for pair in rankings:
                        a, j = pair.tolist()
                        a, j = int(a), int(j)
                        if i == a and output[b, t, j].item() < product_ranges[a, 0].item():
                            ranking_forced = True
                    # Check mutual exclusivity: allow if the value is exactly zero.
                    exclusivity_forced = False
                    for pair in exclusivities:
                        a, k = pair.tolist()
                        a, k = int(a), int(k)
                        if i == a or i == k:
                            if val == 0:
                                exclusivity_forced = True
                    if not (ranking_forced or exclusivity_forced):
                        return False, (
                            f"Batch {b}, time {t}, product {i} has value {val} below its min {min_val} "
                            "and is not forced by a ranking or mutual exclusivity rule."
                        )
    return True, ""

# ---------------------------------------------------------------------
# Fixture: Multiple Constraint Configurations (with diverse, nonnegative min values)
# ---------------------------------------------------------------------
@pytest.fixture(params=[
    {  # Setup 1: Standard constraints – expected explicit output applies here.
        "product_ranges": torch.tensor([[0, 1], [0.5, 2], [1, 3], [1.5, 4]]),
        "exclusivities": torch.tensor([[0, 1], [2, 3]]),
        "product_rankings": torch.tensor([[0, 2], [1, 3]])
    },
    {  # Setup 2: No exclusivity constraints (only range + ranking)
        "product_ranges": torch.tensor([[0, 5], [0, 3], [2, 5], [3, 6]]),
        "exclusivities": torch.tensor([]),  # No exclusivity constraints
        "product_rankings": torch.tensor([[1, 2], [0, 3]])
    },
    {  # Setup 3: More restrictive range constraints
        "product_ranges": torch.tensor([[0.2, 0.5], [0.1, 1.2], [0.5, 2], [1.0, 2.5]]),
        "exclusivities": torch.tensor([[1, 3]]),  # Only one exclusivity rule
        "product_rankings": torch.tensor([[0, 1], [2, 3]])  # Modified ranking rules
    },
    {  # Setup 4: Looser constraints (wider range limits)
        "product_ranges": torch.tensor([[0, 10], [0, 8], [0, 7], [1, 12]]),
        "exclusivities": torch.tensor([[0, 2], [1, 3]]),
        "product_rankings": torch.tensor([[0, 1]])  # Only one ranking constraint
    }
])
def sample_constraints(request):
    """Provides different constraint configurations for testing."""
    return request.param

@pytest.fixture
def constraint_module(sample_constraints):
    """Creates an instance of ConstraintModule for testing."""
    return ConstraintModule(
        number_of_products=4,
        product_ranges=sample_constraints["product_ranges"],
        exclusivities=sample_constraints["exclusivities"],
        product_rankings=sample_constraints["product_rankings"]
    )

# ---------------------------------------------------------------------
# Test 1: Full Constraints Check (Dynamically Adjusted)
# ---------------------------------------------------------------------
def test_constraints_satisfied(constraint_module, sample_constraints):
    """Ensures the final output obeys all constraints over multiple timesteps."""
    input_tensor = torch.tensor([
        [[2.0, 2.5, 3.0, 4.0, 1.0, 0.5, 1.5, 2.2],
         [0.5, 3.0, 2.0, 5.0, 0.9, 1.5, 2.0, 2.5]]
    ])  # Shape: (batch=1, seq=2, products=8)
    output = constraint_module(input_tensor)
    passed, msg = check_range_with_exceptions(
        output,
        sample_constraints["product_ranges"],
        sample_constraints["exclusivities"],
        sample_constraints["product_rankings"]
    )
    assert passed, msg

    exclusivities = sample_constraints["exclusivities"]
    if exclusivities.numel() > 0:
        for pair in exclusivities:
            a, b_idx = pair.tolist()
            a, b_idx = int(a), int(b_idx)
            prod_a = output[..., a]
            prod_b = output[..., b_idx]
            assert (prod_a * prod_b == 0).all(), f"Mutual exclusivity failed for products ({a}, {b_idx})"

    rankings = sample_constraints["product_rankings"]
    if rankings.numel() > 0:
        for pair in rankings:
            a, b_idx = pair.tolist()
            a, b_idx = int(a), int(b_idx)
            assert (output[..., a] <= output[..., b_idx]).all(), f"Ranking constraint failed for ({a} ≤ {b_idx})"

# ---------------------------------------------------------------------
# Test 2: Randomized Input Tests (Dynamically Adjusted)
# ---------------------------------------------------------------------
def test_random_inputs_satisfy_constraints(constraint_module, sample_constraints, K=100):
    """Generate random inputs and ensure all outputs satisfy constraints over multiple timesteps."""
    for _ in range(K):
        batch_size = 4
        sequence_length = 5
        num_products = 4
        input_tensor = torch.rand(batch_size, sequence_length, 2 * num_products) * 10
        output = constraint_module(input_tensor)
        passed, msg = check_range_with_exceptions(
            output,
            sample_constraints["product_ranges"],
            sample_constraints["exclusivities"],
            sample_constraints["product_rankings"]
        )
        assert passed, msg

        exclusivities = sample_constraints["exclusivities"]
        if exclusivities.numel() > 0:
            for pair in exclusivities:
                a, b_idx = pair.tolist()
                a, b_idx = int(a), int(b_idx)
                assert (output[..., a] * output[..., b_idx] == 0).all(), f"Mutual exclusivity failed for ({a}, {b_idx})"

        rankings = sample_constraints["product_rankings"]
        if rankings.numel() > 0:
            for pair in rankings:
                a, b_idx = pair.tolist()
                a, b_idx = int(a), int(b_idx)
                assert (output[..., a] <= output[..., b_idx]).all(), f"Ranking constraint failed for ({a} ≤ {b_idx})"

# ---------------------------------------------------------------------
# Test 3: Explicit Expected Output Verification
# ---------------------------------------------------------------------
def test_explicit_expected_output(constraint_module, sample_constraints):
    """
    Checks that the output exactly matches a pre-calculated expected value.
    
    This test applies only to Setup 1:
      product_ranges = [[0, 1], [0.5, 2], [1, 3], [1.5, 4]]
      exclusivities = [[0, 1], [2, 3]]
      product_rankings = [[0, 2], [1, 3]]
    
    For the sample input below, the expected output (manually calculated) is:
        [[[0.0, 0.5, 0.0, 4.0],
          [0.0, 0.5, 0.0, 3.0]]]
    """
    expected_ranges = torch.tensor([[0, 1], [0.5, 2], [1, 3], [1.5, 4]])
    if not torch.allclose(sample_constraints["product_ranges"].float(), expected_ranges.float()):
        pytest.skip("Explicit expected output test only applies to Setup 1.")
    
    input_tensor = torch.tensor([
        [[1.5, 0.3, 3.0, 4.0, 0.9, 1.5, 2.0, 2.5],
         [2.0, 0.5, 3.5, 3.0, 1.2, 2.5, 1.0, 3.5]]
    ])
    final_output = constraint_module(input_tensor)
    expected_output = torch.tensor([
        [[0.0, 0.5, 0.0, 4.0],
         [0.0, 0.5, 0.0, 3.0]]
    ])
    assert torch.allclose(final_output, expected_output), (
        f"Explicit expected output mismatch!\nExpected:\n{expected_output}\nGot:\n{final_output}"
    )

# ---------------------------------------------------------------------
# Test 4: Giant Random Inputs Piecemeal
# ---------------------------------------------------------------------
def test_giant_random_inputs_piecemeal(K=100):
    """
    Generates giant random inputs with massive batch size, sequence length, and number of products,
    as well as randomly generated constraint parameters. Instead of running the full constraint module
    at once, it runs each constraint (range, mutual exclusivity, ranking) in sequence, checking that 
    the constraints are satisfied after each stage.
    """
    for _ in range(K):
        # Generate random dimensions (massive but tuned to avoid memory issues)
        batch_size = random.randint(64, 128)
        sequence_length = random.randint(64, 128)
        num_products = random.randint(50, 100)

        # Generate random product_ranges: for each product, min in [0,5], range length in [1,10]
        product_ranges_list = []
        for _ in range(num_products):
            min_val = random.uniform(0, 5)
            range_len = random.uniform(1, 10)
            product_ranges_list.append([min_val, min_val + range_len])
        product_ranges = torch.tensor(product_ranges_list)

        # Randomly generate exclusivity constraints: each item appears in at most one pair.
        n_excl = random.randint(0, num_products // 2)
        exclusivity_pairs = []
        available_excl = list(range(num_products))
        for _ in range(n_excl):
            if len(available_excl) < 2:
                break
            pair = random.sample(available_excl, 2)
            exclusivity_pairs.append(pair)
            available_excl.remove(pair[0])
            available_excl.remove(pair[1])
        if exclusivity_pairs:
            exclusivities = torch.tensor(exclusivity_pairs)
        else:
            exclusivities = torch.empty((0, 2), dtype=torch.long)

        # Randomly generate ranking constraints: each item appears in at most one ranking pair.
        n_rank = random.randint(0, num_products // 2)
        ranking_pairs = []
        available_rank = list(range(num_products))
        for _ in range(n_rank):
            if len(available_rank) < 2:
                break
            pair = random.sample(available_rank, 2)  # Do not sort; order matters.
            ranking_pairs.append(pair)
            available_rank.remove(pair[0])
            available_rank.remove(pair[1])
        if ranking_pairs:
            product_rankings = torch.tensor(ranking_pairs)
        else:
            product_rankings = torch.empty((0, 2), dtype=torch.long)

        # Create the constraint module with these random parameters.
        module = ConstraintModule(
            number_of_products=num_products,
            product_ranges=product_ranges,
            exclusivities=exclusivities,
            product_rankings=product_rankings
        )

        # Generate giant random input: shape (batch_size, sequence_length, 2*num_products)
        input_tensor = torch.rand(batch_size, sequence_length, 2 * num_products) * 10

        # --- Piecewise processing ---
        # Step 1: Range Constraint
        range_output = module.range_constraint(input_tensor)
        schedules_range, _ = torch.split(range_output, num_products, dim=-1)
        for i in range(num_products):
            min_val = product_ranges[i, 0].item()
            max_val = product_ranges[i, 1].item()
            prod_values = schedules_range[..., i]
            assert (prod_values >= min_val).all() and (prod_values <= max_val).all(), (
                f"Range constraint failed for product {i}: values outside [{min_val}, {max_val}]."
            )

        # Step 2: Mutual Exclusivity Constraint
        mutual_output = module.mutual_exclusivity_constraint(range_output)
        schedules_mutual, _ = torch.split(mutual_output, num_products, dim=-1)
        for pair in exclusivities:
            a, b = pair.tolist()
            prod_a = schedules_mutual[..., int(a)]
            prod_b = schedules_mutual[..., int(b)]
            assert (prod_a * prod_b == 0).all(), f"Mutual exclusivity failed for products ({a}, {b})."

        # Step 3: Ranking Constraint
        ranking_output = module.ranking_constraint(mutual_output)
        schedules_ranking, _ = torch.split(ranking_output, num_products, dim=-1)
        for pair in product_rankings:
            a, b = pair.tolist()
            prod_a = schedules_ranking[..., int(a)]
            prod_b = schedules_ranking[..., int(b)]
            assert (prod_a <= prod_b).all(), f"Ranking constraint failed for products ({a} ≤ {b})."

        # Final check: Full module output should match piecewise output.
        full_output = module(input_tensor)
        assert torch.allclose(schedules_ranking, full_output), "Piecewise output does not match full module output."

# ---------------------------------------------------------------------
# Test 5: Differentiability Test
# ---------------------------------------------------------------------
def test_module_differentiability(constraint_module, K=10):
    """
    Verifies that the ConstraintModule is differentiable.
    This test creates an input tensor with requires_grad=True, passes it through the module,
    computes a simple loss (sum of outputs), and ensures that gradients are computed with respect to the input.
    """
    batch_size = 2
    sequence_length = 3
    num_products = 4
    for _ in range(K):
        # Create an input tensor with gradients enabled.
        input_tensor = torch.rand(batch_size, sequence_length, 2 * num_products, requires_grad=True)
        output = constraint_module(input_tensor)
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None, "Input tensor gradient is None; module is not differentiable."
        grad_norm = input_tensor.grad.norm()
        print(f"k={_}, grad norm: {grad_norm}")

if __name__ == "__main__":
    pytest.main()