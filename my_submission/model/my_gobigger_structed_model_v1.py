import torch
import torch.nn as nn
from ding.torch_utils import MLP, get_lstm, Transformer
from ding.model import DiscreteHead
from ding.utils import list_split

class RelationGCN(nn.Module):

    def __init__(
            self,
            hidden_shape: int,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(RelationGCN, self).__init__()
        # activation
        self.act = activation
        # layers
        self.clone_with_thorn_relation_layers = MLP(
            hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.clone_with_team_relation_layers = MLP(
            hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.clone_with_enemy_relation_layers = MLP(
            hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.agg_relation_layers = MLP(
            5 * hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )

    def forward(self, clone_with_food_relation, clone_with_thorn_relation, clones, clone_with_team_relation, clone_with_enemy_relation, thorn_mask, clone_mask):
        b, t, c = clones.shape[0], clone_with_thorn_relation.shape[2], clones.shape[1]
        # encode thorn relation
        clone_with_thorn_relation = self.clone_with_thorn_relation_layers(clone_with_thorn_relation) * thorn_mask.view(b, 1, t, 1)  # [b,n_clone,n_thorn,c]
        clone_with_thorn_relation = clone_with_thorn_relation.max(2).values # [b,n_clone,c]

        # encode clone with team relation
        clone_with_team_relation = self.clone_with_team_relation_layers(clone_with_team_relation) * clone_mask.view(b, 1, c, 1) # [b,n_clone,n_clone,c]
        clone_with_team_relation = clone_with_team_relation.max(2).values # [b,n_clone,c]

        # encode clone with enemy relation
        clone_with_enemy_relation = self.clone_with_enemy_relation_layers(clone_with_enemy_relation) * clone_mask.view(b, 1, c, 1) # [b,n_clone,n_clone,c]
        clone_with_enemy_relation = clone_with_enemy_relation.max(2).values # [b,n_clone,c]


        # encode aggregated relation
        agg_relation = torch.cat([clones, clone_with_food_relation, clone_with_thorn_relation, clone_with_team_relation, clone_with_enemy_relation], dim=2)
        clone = self.agg_relation_layers(agg_relation)
        return clone

class Encoder(nn.Module):
    def __init__(
            self,
            scalar_shape: int,
            food_map_shape: int,
            clone_with_food_relation_shape: int,
            clone_with_thorn_relation_shape: int,
            clones_shape: int,
            clone_with_team_relation_shape: int,
            clone_with_enemy_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(Encoder, self).__init__()

        # scalar encoder
        self.scalar_encoder = MLP(
            scalar_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # food encoder
        layers = []
        kernel_size = [5, 3, 1]
        stride = [4, 2, 1]
        shape = [hidden_shape // 4, hidden_shape // 2, hidden_shape]
        input_shape = food_map_shape
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_shape, shape[i], kernel_size[i], stride[i], kernel_size[i] // 2))
            layers.append(activation)
            input_shape = shape[i]
        self.food_encoder = nn.Sequential(*layers)
        # food relation encoder
        self.clone_with_food_relation_encoder = MLP(
            clone_with_food_relation_shape, hidden_shape // 2, hidden_shape, layer_num=2, activation=activation
        )
        # thorn relation encoder
        self.clone_with_thorn_relation_encoder = MLP(
            clone_with_thorn_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone encoder
        self.clones_encoder = MLP(
            clones_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone relation encoder
        self.clone_with_team_relation_encoder = MLP(
            clone_with_team_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        self.clone_with_enemy_relation_encoder = MLP(
            clone_with_enemy_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # gcn
        self.gcn = RelationGCN(
            hidden_shape, activation=activation
        )
        self.agg_encoder = MLP(
            3 * hidden_shape, hidden_shape, encode_shape, layer_num=2, activation=activation
        )
    
    def forward(self, scalar, food_map, clone_with_food_relation, clone_with_thorn_relation, thorn_mask, clones, clone_with_team_relation, clone_with_enemy_relation, clone_mask):
        # encode scalar
        scalar = self.scalar_encoder(scalar) # [b,c]
        # encode food
        food_map = self.food_encoder(food_map) # [b,c,h,w]
        food_map = food_map.reshape(*food_map.shape[:2], -1).max(-1).values # [b,c]
        # encode food relation
        clone_with_food_relation = self.clone_with_food_relation_encoder(clone_with_food_relation) # [b,c]
        # encode thorn relation
        clone_with_thorn_relation = self.clone_with_thorn_relation_encoder(clone_with_thorn_relation) # [b,n_clone,n_thorn, c]
        # encode clone
        clones = self.clones_encoder(clones) # [b,n_clone,c]
        # encode clone with team relation
        clone_with_team_relation = self.clone_with_team_relation_encoder(clone_with_team_relation) # [b,n_clone,n_clone,c]
        # encode clone with enemy relation
        clone_with_enemy_relation = self.clone_with_enemy_relation_encoder(clone_with_enemy_relation) # [b,n_clone,n_clone,c]

        # aggregate all relation
        clones = self.gcn(clone_with_food_relation, clone_with_thorn_relation, clones, clone_with_team_relation, clone_with_enemy_relation, thorn_mask, clone_mask)
        clones = clones.max(1).values # [b,c]

        return self.agg_encoder(torch.cat([scalar, food_map, clones], dim=1))

class MyGoBiggerHybridActionV1(nn.Module):
    r"""
    Overview:
        The GoBiggerHybridAction model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_encoder``, ``compute_critic``
    """
    def __init__(
            self,
            scalar_shape: int,
            food_map_shape: int,
            clone_with_food_relation_shape: int,
            clone_with_thorn_relation_shape: int,
            clones_shape: int,
            clone_with_team_relation_shape: int,
            clone_with_enemy_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            action_type_shape: int,
            rnn: bool = False,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(MyGoBiggerHybridActionV1, self).__init__()
        self.activation = activation
        self.action_type_shape = action_type_shape
        # encoder
        self.encoder = Encoder(scalar_shape, food_map_shape, clone_with_food_relation_shape, clone_with_thorn_relation_shape, clones_shape, clone_with_team_relation_shape, clone_with_enemy_relation_shape, hidden_shape, encode_shape, activation)
        # head
        self.action_type_head = DiscreteHead(32, action_type_shape, layer_num=2, activation=self.activation)

    def forward(self, inputs):
        scalar = inputs['scalar']
        food_map = inputs['food_map']
        clone_with_food_relation = inputs['clone_with_food_relation']
        clone_with_thorn_relation = inputs['clone_with_thorn_relation']
        thorn_mask = inputs['thorn_mask']
        clones = inputs['clones']
        clone_with_team_relation = inputs['clone_with_team_relation']
        clone_with_enemy_relation = inputs['clone_with_enemy_relation']
        clone_mask = inputs['clone_mask']
        fused_embedding_total = self.encoder(scalar, food_map, clone_with_food_relation, clone_with_thorn_relation, thorn_mask, clones, clone_with_team_relation, clone_with_enemy_relation, clone_mask)
        B = inputs['batch']
        A = inputs['player_num_per_team']

        action_type_logit = self.action_type_head(fused_embedding_total)['logit']  # B, M, action_type_size
        action_type_logit = action_type_logit.reshape(B, A, *action_type_logit.shape[1:])

        result = {
            'logit': action_type_logit,
        }
        return result