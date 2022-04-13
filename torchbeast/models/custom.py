import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate

from rl.experiments.pathways.models import ModularPolicy2

from torchbeast.core.popart import PopArtLayer

class MonobeastMP2(torch.nn.Module):
    def __init__(
            self,
            observation_shape,  # not used in this architecture
            num_actions,
            num_tasks=1,
            use_lstm=False,
            use_popart=False,
            reward_clipping="abs_one",
            **_):
        super().__init__()

        self.num_actions = num_actions
        self.num_tasks = num_tasks
        self.reward_clipping = reward_clipping

        assert use_lstm, "MonobeastMP2 only supports LSTM networks"
        assert observation_shape[1] == observation_shape[2] == 84, "MonobeastMP2 only supports 84x84 frames"

        baseline_output_size = 256
        self.net = ModularPolicy2(
            inputs = {
                'frame': {
                    'type': 'GreyscaleImageInput',
                    'config': {
                        'in_channels': observation_shape[0],
                    }
                },
                'reward': {
                    'type': 'ScalarInput',
                },
                'done': {
                    'type': 'ScalarInput',
                },
                #'action': {
                #    'type': 'DiscreteInput',
                #    'config': {
                #        'input_size': num_actions,
                #    }
                #},
            },
            outputs = {
                'policy_logits': {
                    'type': 'LinearOutput',
                    'config': {
                        'output_size': num_actions,
                    }
                },
                'baseline': {
                    'type': 'LinearOutput',
                    'config': {
                        'output_size': baseline_output_size,
                    }
                },
            },
            input_size=512,
            key_size=512,
            value_size=512,
            num_heads=8,
            ff_size=1024,
            recurrence_type='RecurrentAttention9',
        )
        self.baseline = PopArtLayer(baseline_output_size, num_tasks if use_popart else 1)

    def forward(self, inputs, core_state, run_to_conv=-1, stochastic=False):
        if run_to_conv >= 0:
            x = inputs
        else:
            x = inputs["frame"]

        T, B, *_ = x.shape
        #x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        clipped_reward = None
        if self.reward_clipping == "abs_one":
            clipped_reward = torch.clamp(inputs["reward"], -1, 1)
        elif self.reward_clipping == "none":
            clipped_reward = inputs["reward"]
        else:
            raise ValueError("Unknown reward clipping method: {}".format(self.reward_clipping))

        output = []
        for i in range(T):
            o = self.net({
                    'frame': x[i,...], 
                    'reward': clipped_reward[i,...].unsqueeze(1),
                    'done': inputs['done'][i,...].unsqueeze(1),
            }, core_state)
            output.append(o)
            core_state = o['hidden'] # type: ignore
        output = default_collate(output)

        baseline, normalized_baseline = self.baseline(output['baseline'].view(T * B, -1)) # type: ignore

        if self.training or stochastic:
            action = torch.multinomial(F.softmax(output['policy_logits'].view(T * B, -1), dim=1), num_samples=1) # type: ignore
        else:
            # Don't sample when testing.
            action = torch.argmax(output['policy_logits'].view(T * B, -1), dim=2) # type: ignore

        return ({
            'policy_logits': output['policy_logits'].view(T, B, self.num_actions), # type: ignore
            'baseline': baseline.view(T, B, self.num_tasks),
            'normalized_baseline': normalized_baseline.view(T, B, self.num_tasks),
            'action': action.view(T, B, 1),
        }, core_state)

    def initial_state(self, batch_size):
        return self.net.init_hidden(batch_size)
