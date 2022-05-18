import torch


class Criterion(torch.nn.Module):

    def __init__(self, losses):
        super().__init__()
        self.losses = losses
        

    def loss_value(self, prediction_dict: dict[str, torch.Tensor], target_dict: dict[str, torch.Tensor]):
        """Score for the board value; doing this MSE style"""
        mse_loss_handle = torch.nn.MSELoss()
        loss = {}
        
        for key in prediction_dict:
            
            # Isolate wanted keys 
            key_str_list = key.split('_')
            if key_str_list[0] == 'value':
                
                # compute loss
                loss[key] = mse_loss_handle(torch.tanh(prediction_dict[key]),
                                            target_dict['value'])

        return loss

    def loss_policy(self, prediction_dict: dict[str, torch.Tensor], target_dict: dict[str, torch.Tensor]):
        """Cross entropy loss; if there is a win vector, only counts probability vectors that the win flag is true."""

        loss_ce = torch.nn.CrossEntropyLoss()
        loss = {}
        
        for key in prediction_dict:
            
            # Isolate wanted keys 
            key_str_list = key.split('_')
            if key_str_list[0] == 'policy':
                
                # Check if target includes the win key
                if 'win' in target_dict.keys():
                    
                    loss[key] = 0
                    loss[key] += loss_ce(prediction_dict[key][target_dict['win'], :], 
                                                   target_dict['policy'][target_dict['win'], :])
                    
        return loss

    def get_loss(self, loss, prediction_dict: dict[str, torch.Tensor], target_dict: dict[str, torch.Tensor]):
        loss_map = {
            'loss_policy': self.loss_policy,
            'loss_value': self.loss_value
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](prediction_dict, target_dict)

    def forward(self, prediction_dict: dict[str, torch.Tensor], target_dict: dict[str, torch.Tensor]):

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, prediction_dict, target_dict))

        return losses
