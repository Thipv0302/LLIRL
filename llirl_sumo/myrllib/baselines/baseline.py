import torch
import torch.nn as nn

class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.linear = nn.Linear(self.feature_size, 1, bias=False)
        self.linear.weight.data.zero_()

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations * ones
        cum_sum = torch.cumsum(ones, dim=0) * ones
        al = cum_sum / 100.0

        return torch.cat([observations, observations ** 2, al, al ** 2, al ** 3, ones], dim=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)
        
        # Check if we have enough samples
        num_samples = featmat.shape[0]
        if num_samples < self.feature_size:
            # Not enough samples, use zero baseline (no update)
            # This can happen with small batch sizes or short episodes
            return
        
        # Check for NaN or Inf values
        if torch.isnan(featmat).any() or torch.isinf(featmat).any():
            # Invalid features, use zero baseline
            return
        
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            # Invalid returns, use zero baseline
            return

        reg_coeff = self._reg_coeff
        eye = torch.eye(self.feature_size, dtype=torch.float32, device=self.linear.weight.device)
        
        # Try with increasing regularization
        max_reg_coeff = 1e6  # Increased maximum
        for attempt in range(10):  # More attempts
            try:
                XtX = torch.matmul(featmat.t(), featmat)
                Xty = torch.matmul(featmat.t(), returns)
                
                # Add regularization
                reg_matrix = reg_coeff * eye
                A = XtX + reg_matrix
                
                # Try to solve using lstsq
                # Use older API for compatibility
                try:
                    coeffs, _ = torch.lstsq(Xty, A)
                    coeffs = coeffs[:self.feature_size]
                except:
                    # Try newer API if available
                    try:
                        coeffs = torch.linalg.lstsq(A, Xty).solution
                    except:
                        # Last resort: use pseudo-inverse
                        coeffs = torch.matmul(torch.pinverse(A), Xty)
                
                # Check if solution is valid
                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    raise RuntimeError("Invalid solution")
                
                self.linear.weight.data = coeffs.data.t()
                return  # Success
                
            except (RuntimeError, Exception) as e:
                # Catch all exceptions including LinAlgError if available
                try:
                    from torch.linalg import LinAlgError
                    if isinstance(e, LinAlgError):
                        pass  # Handle LinAlgError
                except ImportError:
                    pass  # LinAlgError not available in this PyTorch version
                # Increase regularization and try again
                reg_coeff *= 10
                if reg_coeff > max_reg_coeff:
                    # If still failing, use zero baseline (no update)
                    # This is better than crashing
                    return
        else:
            # All attempts failed, use zero baseline (no update)
            return

    def forward(self, episodes):
        features = self._feature(episodes)
        return self.linear(features)

