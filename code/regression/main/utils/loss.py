import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    """
    Wing Loss for robust detection, adapted for tasks with dynamic landmark counts.
    Transitions from logarithmic to linear form outside a defined pixel width, tailored for precision in key point prediction.
    Automatically adjusts to the number of landmarks based on input tensor shape.

    Parameters:
        w (float): Width of the piecewise function's linear region, beyond which errors are considered large.
        epsilon (float): Adjusts the curvature of the logarithmic part, affecting sensitivity to small errors.
    """

    def __init__(self, w=3.0, epsilon=1.5):
        """
        Initializes the WingLoss with specified parameters.
        Parameters:
            w (float): Controls the width of the non-linear region.
            epsilon (float): Defines the curvature of the non-linear region.
        """
        super(WingLoss, self).__init__()
        self.w = torch.tensor(w)  # Convert w to a tensor
        self.epsilon = torch.tensor(epsilon)  # Convert epsilon to a tensor
        self.C = self.w - self.w * torch.log(1.0 + self.w / self.epsilon)

    def forward(self, prediction, target):
        """
        Computes the WingLoss between predicted and target landmark coordinates.

        Parameters:
            prediction (torch.Tensor): Predicted coordinates (batch_size, num_landmarks*2).
                                       Each landmark represented by an (x, y) pair.
            target (torch.Tensor): Ground truth coordinates, same shape as `prediction`.

        Returns:
            torch.Tensor: The loss for each landmark, facilitating detailed analysis and backpropagation.
        """
        y = torch.abs(prediction - target)
        loss = torch.where(y < self.w, self.w * torch.log(1.0 + y / self.epsilon), y - self.C)
        num_landmarks = prediction.size(1) // 2  # Dynamically infer the number of landmarks
        loss_per_coordinate = loss.view(-1, num_landmarks, 2)  # Adjust to the inferred number of landmarks
        mean_loss_per_landmark = torch.mean(loss_per_coordinate, dim=0)
        return mean_loss_per_landmark

class MultifacetedLoss(nn.Module):
    """
    This multifaceted approach ensures the model's focus on key aspects of landmark positioning, including the precise
    delineation of pectoralis muscle and nipple points. 
    
    Parameters:
        w (float): Width of the piecewise function's linear region in Wing Loss.
        epsilon (float): Curvature adjustment for the logarithmic part in Wing Loss.
        alpha (float): Weight for the pec1 loss component.
        beta (float): Weight for the pec2 loss component.
        gamma (float): Weight for the nipple loss component.
    """
    def __init__(self, w=5.0, epsilon=2, alpha=1.0, 
                 beta=1.0, gamma=1.0):
        super(MultifacetedLoss, self).__init__()
        self.wing_loss = WingLoss(w, epsilon)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
 
    def forward(self, prediction, target):
        wing_loss_values = self.wing_loss(prediction, target)
        pec1_loss_value, pec2_loss_value, nipple_loss_value = wing_loss_values.mean(dim=1)
        angle_loss_value = self.angle_loss(prediction, target)

        total_loss = self.alpha * pec1_loss_value + \
                     self.beta * pec2_loss_value + \
                     self.gamma * nipple_loss_value
        
        print(f"pec1_loss: {pec1_loss_value * self.alpha:.5f} * "\
              f"pec2_loss: {pec2_loss_value * self.beta:.5f} * "\
              f"nipple_loss: {nipple_loss_value * self.gamma:.5f} \n"
              f"Total loss: {total_loss:.5f}")
        
        return total_loss
    
