import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class COR(nn.Module):


    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims,
                adj, E1_size, n_items, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_dims = mlp_p1_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.n_items = n_items
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs


        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid


        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_dims = self.mlp_p1_dims[:-1] + [self.mlp_p1_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims


        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p1_dims[:-1], temp_p1_dims[1:])])
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])

        # Eq. for E1*: counterfactual feature-shift generator G(·).
        # E1* = G(E1), modelling potential preference changes caused by contextual variations.
        # Initialised near-identity so training starts from E1* ≈ E1.
        self.feature_generator = nn.Sequential(
            nn.Linear(E1_size, E1_size),
            nn.Tanh(),
            nn.Linear(E1_size, E1_size),
        )
        # Eq. 21: item embeddings e_t for counterfactual interaction reweighting.
        # w_t = softmax_t(-||e_t - E1*||^2). padding_idx=0 keeps item 0 (padding) neutral.
        self.interaction_embeddings = nn.Embedding(n_items, E1_size, padding_idx=0)

        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)

        self.init_weights()

    def generate_E1_star(self, E1):
        """E1* = G(E1): counterfactual feature state."""
        return self.feature_generator(E1)

    def generate_D_star(self, D, E1_star):
        """Counterfactual interaction representation (Eq. 19-21).

        D* = Σ_t w_t d_t,  w_t = softmax_t(-||e_t - E1*||^2).
        Interactions consistent with E1* receive higher weight; outdated interactions are down-weighted.
        Returns a reweighted interaction vector with the same shape as D.
        A large finite value (1e9) is used for non-interacted items instead of +inf so that all-zero
        rows (users without interactions) produce a uniform softmax instead of NaN.
        """
        item_emb = self.interaction_embeddings.weight  # [n_items, E1_size]
        # d2[b, t] = ||e_t - E1*[b]||^2 (squared Euclidean distance)
        d2 = torch.cdist(E1_star, item_emb)            # [B, n_items]
        # only interacted items participate in the softmax
        mask = (D == 0)                                 # [B, n_items]
        d2 = d2.masked_fill(mask, 1e9)                  # large finite (not inf) -> avoids NaN for all-zero rows
        w = torch.softmax(-d2, dim=-1)                  # [B, n_items]
        # D* = Σ_t w_t d_t  (reweighted interaction vector; 0 for non-interacted)
        D_star = w * D
        return D_star

    def forward(self, D, E1, CI=0):
        """Forward pass computing both factual (P) and counterfactual (P*) predictions.

        Returns: P, P_star, mu, logvar (factual E2), Z2 (Z_stable), Z2_star (Z_stable*), reg_loss
        """
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)

        # ---- Factual path ----
        # Eq. 15: E2 ~ N(μ_E2, Σ_E2) via MLP_q([D, E1])
        encoder_input = torch.cat((D, E1), 1)
        mu, logvar = self.encode(encoder_input)
        E2 = self.reparameterize(mu, logvar)
        # Eq. 18 (factual): Z1 ~ N(μ_Z1, Σ_Z1) via MLP_p1([E1, E2])
        Z1_mu, Z1_logvar = self.decode_p1(E1, E2)
        # Eq. 16 (factual): Z2 ~ N(μ_Z2, Σ_Z2) via MLP_p2(E2)  (Z_stable)
        Z2_mu, Z2_logvar = self.decode_p2(E2)
        Z1 = self.reparameterize(Z1_mu, Z1_logvar)
        Z2 = self.reparameterize(Z2_mu, Z2_logvar)
        # Eq. 22 (factual): P = f(Z_stable) = f([Z1, Z2])
        P = self.decode_p3(Z1, Z2)

        # ---- Counterfactual path ----
        # E1* = G(E1)
        E1_star = self.generate_E1_star(E1)
        # Eq. 19-21: D* = Σ_t w_t d_t
        D_star = self.generate_D_star(D, E1_star)
        # Eq. 17: E2* ~ N(μ_E2*, Σ_E2*) via MLP_q([D*, E1*])
        E2_star_input = torch.cat((D_star, E1_star), 1)
        mu_star, logvar_star = self.encode(E2_star_input)
        E2_star = self.reparameterize(mu_star, logvar_star)
        # Eq. 18 (counterfactual): Z1* ~ N(μ_Z1*, Σ_Z1*) via MLP_p1([E1*, E2*])
        Z1_star_mu, Z1_star_logvar = self.decode_p1(E1_star, E2_star)
        # Eq. 16 (counterfactual): Z2* ~ N(μ_Z2*, Σ_Z2*) via MLP_p2(E2*)  (Z_stable*)
        Z2_star_mu, Z2_star_logvar = self.decode_p2(E2_star)
        Z1_star = self.reparameterize(Z1_star_mu, Z1_star_logvar)
        Z2_star = self.reparameterize(Z2_star_mu, Z2_star_logvar)
        # Eq. 22 (counterfactual): P* = f(Z_stable*) = f([Z1*, Z2*])
        P_star = self.decode_p3(Z1_star, Z2_star)

        reg_loss = self.reg_loss()
        return P, P_star, mu, logvar, Z2, Z2_star, reg_loss

    def encode(self, encoder_input):

        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h)
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_p1(self, E1_input, E2_input):
        """MLP_p1: [E1_input, E2_input] -> (Z1_mu, Z1_logvar).

        Factual: Z1 from [E1, E2]; counterfactual: Z1* from [E1*, E2*].
        """
        h = torch.cat((E1_input, E2_input), 1)
        for i, layer in enumerate(self.mlp_p1_layers):
            h = layer(h)
            if i != len(self.mlp_p1_layers) - 1:
                h = self.act_function(h)
            else:
                Z1_mu = h[:, :self.mlp_p1_dims[-1]]
                Z1_logvar = h[:, self.mlp_p1_dims[-1]:]
        return Z1_mu, Z1_logvar

    def decode_p2(self, E2_input):
        """MLP_p2: E2_input -> (Z2_mu, Z2_logvar).

        Factual: Z2 = p2(E2) (Z_stable); counterfactual: Z2* = p2(E2*) (Z_stable*).
        """
        h = E2_input
        for i, layer in enumerate(self.mlp_p2_layers):
            h = layer(h)
            if i != len(self.mlp_p2_layers) - 1:
                h = self.act_function(h)
            else:
                Z2_mu = h[:, :self.mlp_p2_dims[-1]]
                Z2_logvar = h[:, self.mlp_p2_dims[-1]:]
        return Z2_mu, Z2_logvar

    def decode_p3(self, Z1, Z2):
        """MLP_p3: [Z1, Z2] -> item scores. P = f(Z_stable), P* = f(Z_stable*)."""
        user_preference = torch.cat((Z1, Z2), 1)
        h = user_preference
        for i, layer in enumerate(self.mlp_p3_layers):
            h = layer(h)
            if i != len(self.mlp_p3_layers) - 1:
                h = self.act_function(h)
        return h

    def init_weights(self):

        for layer in self.mlp_q_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p1_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p2_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p3_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        # Feature-shift generator G: initialise near-identity so E1* ≈ E1 at start.
        for layer in self.feature_generator:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0.0, 0.001)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        # Make the output linear layer of G approximate identity (add to input).
        linear_layers = [l for l in self.feature_generator if isinstance(l, nn.Linear)]
        if len(linear_layers) >= 2:
            linear_layers[-1].weight.data.add_(torch.eye(self.E1_size))
        # Interaction embeddings e_t: small normal init.
        self.interaction_embeddings.weight.data.normal_(0.0, 0.001)
        if self.interaction_embeddings.padding_idx is not None:
            self.interaction_embeddings.weight.data[self.interaction_embeddings.padding_idx].zero_()

    def reg_loss(self):

        reg_loss = 0

        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)

        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)

        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)

        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)

        for name, parm in self.feature_generator.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)

        for name, parm in self.interaction_embeddings.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)

        return reg_loss


class COR_G(nn.Module):


    def __init__(self, mlp_q_dims, mlp_p1_1_dims, mlp_p1_2_dims, mlp_p2_dims, mlp_p3_dims,
                 item_feature, adj, E1_size, n_items, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR_G, self).__init__()

        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_1_dims = mlp_p1_1_dims
        self.mlp_p1_2_dims = mlp_p1_2_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.n_items = n_items
        self.Z1_size = adj.size(0)
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs


        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid


        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim],
                                               requires_grad=True).cuda()


        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_1_dims = self.mlp_p1_1_dims
        temp_p1_2_dims = self.mlp_p1_2_dims[:-1] + [self.mlp_p1_2_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims


        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                              d_in, d_out in zip(temp_p1_1_dims[:-1], temp_p1_1_dims[1:])])


        self.mlp_p1_2_layers = [(torch.randn([self.Z1_size, d_in, d_out], requires_grad=True)).cuda() for
                                d_in, d_out in zip(temp_p1_2_dims[:-1], temp_p1_2_dims[1:])]


        for i, matrix in enumerate(self.mlp_p1_2_layers):
            temp = torch.unsqueeze(matrix, 0) if i == 0 else torch.cat((temp, torch.unsqueeze(matrix, 0)), 0)
        self.mlp_p1_2_layers = nn.Parameter(temp)


        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])

        # Eq. for E1*: counterfactual feature-shift generator G(·).
        self.feature_generator = nn.Sequential(
            nn.Linear(E1_size, E1_size),
            nn.Tanh(),
            nn.Linear(E1_size, E1_size),
        )
        # Eq. 21: item embeddings e_t for counterfactual reweighting.
        self.interaction_embeddings = nn.Embedding(n_items, E1_size, padding_idx=0)

        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)
        self.init_weights()

    def generate_E1_star(self, E1):
        """E1* = G(E1): counterfactual feature state."""
        return self.feature_generator(E1)

    def generate_D_star(self, D, E1_star):
        """Counterfactual interaction representation (Eq. 19-21). D* = Σ_t w_t d_t."""
        item_emb = self.interaction_embeddings.weight  # [n_items, E1_size]
        d2 = torch.cdist(E1_star, item_emb)            # [B, n_items]
        mask = (D == 0)                                 # [B, n_items]
        d2 = d2.masked_fill(mask, 1e9)                  # large finite (not inf) -> avoids NaN for all-zero rows
        w = torch.softmax(-d2, dim=-1)                  # [B, n_items]
        D_star = w * D
        return D_star

    def forward(self, D, E1, CI=0):
        """Forward pass computing both factual (P) and counterfactual (P*) predictions.

        Returns: P, P_star, mu, logvar (factual E2), Z2 (Z_stable), Z2_star (Z_stable*), reg_loss
        """
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)

        # ---- Factual path ----
        encoder_input = torch.cat((D, E1), 1)
        mu, logvar = self.encode(encoder_input)
        E2 = self.reparameterize(mu, logvar)
        Z1_mu, Z1_logvar = self.decode_p1(E1, E2)
        Z2_mu, Z2_logvar = self.decode_p2(E2)
        Z1 = self.reparameterize(Z1_mu, Z1_logvar)
        Z2 = self.reparameterize(Z2_mu, Z2_logvar)
        P = self.decode_p3(Z1, Z2)

        # ---- Counterfactual path ----
        E1_star = self.generate_E1_star(E1)
        D_star = self.generate_D_star(D, E1_star)
        E2_star_input = torch.cat((D_star, E1_star), 1)
        mu_star, logvar_star = self.encode(E2_star_input)
        E2_star = self.reparameterize(mu_star, logvar_star)
        Z1_star_mu, Z1_star_logvar = self.decode_p1(E1_star, E2_star)
        Z2_star_mu, Z2_star_logvar = self.decode_p2(E2_star)
        Z1_star = self.reparameterize(Z1_star_mu, Z1_star_logvar)
        Z2_star = self.reparameterize(Z2_star_mu, Z2_star_logvar)
        P_star = self.decode_p3(Z1_star, Z2_star)

        reg_loss = self.reg_loss()
        return P, P_star, mu, logvar, Z2, Z2_star, reg_loss

    def encode(self, encoder_input):
        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h)
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_p1(self, E1_input, E2_input):
        """Graph-structured MLP_p1: [E1_input, E2_input] -> (Z1_mu, Z1_logvar).

        Factual: Z1 from [E1, E2]; counterfactual: Z1* from [E1*, E2*].
        """
        h_p1 = torch.cat((E1_input, E2_input), 1)

        h_p1 = torch.unsqueeze(h_p1, -1)
        for i, layer in enumerate(self.mlp_p1_1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_1_layers) - 1:
                h_p1 = self.act_function(h_p1)

        h_p1 = torch.matmul(self.adj, h_p1)
        h_p1 = torch.unsqueeze(h_p1, 2)
        for i, matrix in enumerate(self.mlp_p1_2_layers):
            h_p1 = torch.matmul(h_p1, matrix)
            if i != len(self.mlp_p1_2_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                h_p1 = torch.squeeze(h_p1)
                Z1_mu = torch.squeeze(h_p1[:, :, :self.mlp_p1_2_dims[-1]])
                Z1_logvar = torch.squeeze(h_p1[:, :, self.mlp_p1_2_dims[-1]:])
        return Z1_mu, Z1_logvar

    def decode_p2(self, E2_input):
        """MLP_p2: E2_input -> (Z2_mu, Z2_logvar).

        Factual: Z2 = p2(E2) (Z_stable); counterfactual: Z2* = p2(E2*) (Z_stable*).
        """
        h_p2 = E2_input
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]
        return Z2_mu, Z2_logvar

    def decode_p3(self, Z1, Z2):
        """MLP_p3: [Z1, Z2] -> item scores. P = f(Z_stable), P* = f(Z_stable*)."""
        user_preference = torch.cat((Z1, Z2), 1)

        h_p3 = user_preference
        for i, layer in enumerate(self.mlp_p3_layers):
            h_p3 = layer(h_p3)
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)
        return h_p3

    def init_weights(self):

        for layer in self.mlp_q_layers:

            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p1_1_layers:

            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            layer.bias.data.normal_(0.0, 0.001)

        for matrix in self.mlp_p1_2_layers:

            size = matrix.data.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            matrix.data.normal_(0.0, std)

        for layer in self.mlp_p2_layers:

            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p3_layers:

            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            layer.bias.data.normal_(0.0, 0.001)

        # Feature-shift generator G: initialise near-identity so E1* ≈ E1 at start.
        for layer in self.feature_generator:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0.0, 0.001)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        linear_layers = [l for l in self.feature_generator if isinstance(l, nn.Linear)]
        if len(linear_layers) >= 2:
            linear_layers[-1].weight.data.add_(torch.eye(self.E1_size))
        # Interaction embeddings e_t: small normal init.
        self.interaction_embeddings.weight.data.normal_(0.0, 0.001)
        if self.interaction_embeddings.padding_idx is not None:
            self.interaction_embeddings.weight.data[self.interaction_embeddings.padding_idx].zero_()

    def reg_loss(self):

        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        for name, parm in self.feature_generator.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        for name, parm in self.interaction_embeddings.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        return reg_loss

def loss_function(P_star, x, mu, logvar, z_stable, z_stable_star, reg_loss, anneal, lambda1=1.0, lambda2=0.0, lambda3=1.0):
    """Composite loss for unstable users (Eq. 20, 21, 23, 25).

    L_Unstable = L_rec + λ1 * anneal * L_KL + λ2 * L_reg + λ3 * L_CONS

    - L_rec: cross-entropy on the counterfactual prediction P* (Eq. 23)
    - L_KL: KL divergence on the factual E2 (Eq. 20), annealed by the KL annealing schedule
    - L_reg: L2 regularization (Eq. 21)
    - L_CONS: counterfactual consistency ||Z_stable - Z_stable*||^2 (Eq. 19)
    """
    # Eq. 23: L_rec = -Σ y_ui log P*_ui (cross-entropy on the counterfactual prediction)
    BCE = -torch.mean(torch.sum(F.log_softmax(P_star, 1) * x, -1))

    # Eq. 20: L_KL on the factual E2
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # Eq. 19: L_CONS = ||Z_stable - Z_stable*||^2
    CONS = torch.mean(torch.sum((z_stable - z_stable_star) ** 2, dim=1))

    # Eq. 25: L_Unstable = L_rec + λ1 * L_KL + λ2 * L_reg + λ3 * L_CONS
    return BCE + lambda1 * anneal * KLD + lambda2 * reg_loss + lambda3 * CONS