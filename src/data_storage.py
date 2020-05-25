from dataclasses import dataclass
import torch


@dataclass
class StoreState:
    """
    Class for storing state variables

    *n_prefix* : current state
    p_* : prime state
    """
    position: torch.Tensor = torch.zeros((1, 4, 4, 15))
    speed: torch.Tensor = torch.zeros((1, 4, 4, 15))
    tl: torch.Tensor = torch.zeros((1, 1, 4))
    p_position: torch.Tensor = torch.zeros((1, 4, 4, 15))
    p_speed: torch.Tensor = torch.zeros((1, 4, 4, 15))
    p_tl: torch.Tensor = torch.zeros((1, 1, 4))

    action: 'typing.Any' = 0
    reward: 'typing.Any' = 0
    done_mask: float = 1.0

    def concat(self):
        self.position = torch.cat(self.position)
        self.speed = torch.cat(self.speed)
        self.tl = torch.cat(self.tl)
        self.p_position = torch.cat(self.p_position)
        self.p_speed = torch.cat(self.p_speed)
        self.p_tl = torch.cat(self.p_tl)

        return (self.position, self.speed, self.tl), \
               (self.p_position, self.p_speed, self.p_tl)

    @property
    def as_tuple(self):
        return (self.position, self.speed, self.tl)



