import torch
from torch import nn

class RangeConstraint(nn.Module):
    """
    PyTorch module that enforces range constraints on product schedules.

    Attributes:
        product_ranges (torch.Tensor): A tensor of shape (number_of_products, 2)
                                      specifying the min and max allowable values
                                      for each product.

    Example:
        product_ranges = torch.tensor([
            [0, 1],      # Product 0 in range [0,1]
            [-10, 20],   # Product 1 in range [-10,20]
            [9, 10]      # Product 2 in range [9,10]
        ])
    """

    def __init__(self, number_of_products: int, product_ranges: torch.Tensor):
        """
        Args:
            product_ranges (torch.Tensor): Tensor of shape (number_of_products, 2)
                                           specifying min and max values.
        """
        super().__init__()
        self.number_of_products = number_of_products
        self.product_ranges = product_ranges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply range constraints using element-wise clamping.

        Args:
            x (torch.Tensor): Input tensor (from output of transformer model) of shape
                              (batch, sequence_length, 2*number_of_products). x[:, :, :number_of_products]
                              are predicted schedules and x[:, :, :, number_of_products:] are priority values
                              identifying which product to choose in the case of mutual exclusivity.

        Returns:
            torch.Tensor: Tensor with values clamped within the specified ranges of shape (batch_size, sequence_length, 2*number_of_products)
        """
        if self.product_ranges.numel() < 2:
            return x
        schedules, priorities = torch.split(x, self.number_of_products, dim=-1)
        min_vals = self.product_ranges[:, 0].view(1, 1, -1)
        max_vals = self.product_ranges[:, 1].view(1, 1, -1)
        ranged_schedules = torch.clamp(schedules, min_vals, max_vals)
        output = torch.cat([ranged_schedules, priorities], dim=-1)
        return output


class MutualExclusivityConstraint(nn.Module):
    """
    PyTorch module that enforces mutual exclusivity constraints.

    Attributes:
        exclusivities (torch.Tensor): A tensor of shape (num_constraints, 2),
                                      where each row specifies two mutually
                                      exclusive product indices.

    Example:
        exclusivities = torch.tensor([
            [0, 1],  # Product 0 and 1 are mutually exclusive
            [2, 3]   # Product 2 and 3 are mutually exclusive
        ])
    """

    def __init__(self, number_of_products: int, exclusivities: torch.Tensor):
        """
        Args:
            exclusivities (torch.Tensor): A tensor of shape (num_constraints, 2)
                                          defining mutually exclusive products.
        """
        super().__init__()
        self.number_of_products = number_of_products
        self.exclusivities = exclusivities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce mutual exclusivity by zeroing out one of the two conflicting products.

        Args:
            x (torch.Tensor): Input tensor (from output of transformer model) of shape
                              (batch, sequence_length, 2*number_of_products). x[:, :, :number_of_products]
                              are predicted schedules and x[:, :, :, number_of_products:] are priority values
                              identifying which product to choose in the case of mutual exclusivity.
        Returns:
            torch.Tensor: Tensor with mutually exclusive constraints applied of shape (batch_size, sequence_length, 2*number_of_products).
        """
        if self.exclusivities.numel() < 2:
            return x
        schedules, priorities = torch.split(x, self.number_of_products, dim=-1)
        mask = torch.ones_like(schedules).bool()
        mask[:, :, self.exclusivities[:, 0]] = priorities[:, :, self.exclusivities[:, 0]] >= priorities[:, :, self.exclusivities[:, 1]]
        mask[:, :, self.exclusivities[:, 1]] = priorities[:, :, self.exclusivities[:, 1]] > priorities[:, :, self.exclusivities[:, 0]]
        exclusive_schedules = schedules * mask
        output = torch.cat([exclusive_schedules, priorities], dim=-1)
        return output


class RankingConstraint(nn.Module):
    """
    PyTorch module that enforces ranking constraints.

    Attributes:
        product_rankings (torch.Tensor): A tensor of shape (num_constraints, 2),
                                         where each row specifies two products where
                                         the first must always be less than or equal to the second.

    Example:
        product_rankings = torch.tensor([
            [0, 1],  # Product 0 ≤ Product 1
            [1, 2]   # Product 1 ≤ Product 2
        ])
    """

    def __init__(self, number_of_products: int, product_rankings: torch.Tensor):
        """
        Args:
            product_rankings (torch.Tensor): A tensor of shape (num_constraints, 2)
                                             defining ranking constraints.
        """
        super().__init__()
        self.number_of_products = number_of_products
        self.product_rankings = product_rankings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ranking constraints by ensuring the ranked products obey the ordering.

        Args:
            x (torch.Tensor): Input tensor (from output of transformer model) of shape
                              (batch, sequence_length, 2*number_of_products). x[:, :, :number_of_products]
                              are predicted schedules and x[:, :, :, number_of_products:] are priority values
                              identifying which product to choose in the case of mutual exclusivity.

        Returns:
            torch.Tensor: Tensor with ranking constraints enforced of shape (batch_size, sequence_length, 2*number_of_products).
        """
        if self.product_rankings.numel() < 2:
            return x
        schedules, priorities = torch.split(x, self.number_of_products, dim=-1)
        ranked_schedules = schedules.clone()
        ranked_schedules[:, :, self.product_rankings[:, 0]] = torch.min(schedules[:, :, self.product_rankings[:,0]],
                                                              schedules[:, :, self.product_rankings[:, 1]])
        output = torch.cat([ranked_schedules, priorities], dim=-1)
        return output

class ConstraintModule(nn.Module):
    """
    PyTorch module that sequentially applies range, mutual exclusivity, and ranking constraints 
    to the outputs of a transformer model.

    This module ensures that:
      - Product schedules remain within valid ranges.
      - Mutually exclusive products do not have nonzero schedules simultaneously.
      - Certain product schedules always remain less than or equal to others.

    Attributes:
        range_constraint (RangeConstraint): Applies range limits to product schedules.
        mutual_exclusivity_constraint (MutualExclusivityConstraint): Enforces mutual exclusivity constraints.
        ranking_constraint (RankingConstraint): Enforces ranking relationships between product schedules.

    Example:
        constraint_module = ConstraintModule(
            number_of_products=3,
            product_ranges=torch.tensor([[0, 1], [-10, 20], [9, 10]]),
            exclusivities=torch.tensor([[0, 1]]),
            product_rankings=torch.tensor([[0, 2]])
        )
        constrained_output = constraint_module(predicted_output)
    """

    def __init__(self, number_of_products: int, product_ranges: torch.Tensor, 
                 exclusivities: torch.Tensor, product_rankings: torch.Tensor):
        """
        Args:
            number_of_products (int): The number of products in the output schedule.
            product_ranges (torch.Tensor): A tensor of shape (number_of_products, 2) specifying min/max values.
            exclusivities (torch.Tensor): A tensor of shape (num_constraints, 2) defining mutually exclusive products.
            product_rankings (torch.Tensor): A tensor of shape (num_constraints, 2) defining ranking constraints.
        """
        super().__init__()
        self.number_of_products = number_of_products
        self.range_constraint = RangeConstraint(number_of_products, product_ranges)
        self.mutual_exclusivity_constraint = MutualExclusivityConstraint(number_of_products, exclusivities)
        self.ranking_constraint = RankingConstraint(number_of_products, product_rankings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the range, mutual exclusivity, and ranking constraints sequentially.

        Args:
            x (torch.Tensor): Input tensor from the transformer model of shape 
                              (batch_size, sequence_length, 2*number_of_products). 
                              The first half of the last dimension corresponds to predicted schedules,
                              and the second half corresponds to priority values.

        Returns:
            torch.Tensor: The post-processed schedule tensor with constraints applied.
                         Output shape: (batch_size, sequence_length, number_of_products).
        """
        # Apply constraints in a sequential manner
        x = self.range_constraint(x)
        x = self.mutual_exclusivity_constraint(x)
        x = self.ranking_constraint(x)

        # Extract only the schedules from the final processed tensor (drop priorities)
        schedules, _ = torch.split(x, self.number_of_products, dim=-1)
        return schedules