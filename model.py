import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import DiffTransformer
# from Transformer_NumDAE import DiffEncoder, DiffDecoder


class Gate(nn.Module):
    def __init__(self, input_size, output_size, gate_activation=torch.sigmoid):
        super(Gate, self).__init__()
        self.output_size = output_size
        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        if x_ent.shape != x_lit.shape:
            x_ent = x_ent.view(x_lit.shape)

        x = torch.cat([x_ent, x_lit], x_lit.ndimension() - 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1 - gate) * x_ent + gate * g_embedded
        return output


class StructuralGraphAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_edge_types=6, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5
        self.num_edge_types = num_edge_types

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.key_biases = nn.Embedding(self.num_edge_types, self.d_head)
        self.value_biases = nn.Embedding(self.num_edge_types, self.d_head)

        if self.num_edge_types > 5:
            nn.init.zeros_(self.key_biases.weight.data[5])
            nn.init.zeros_(self.value_biases.weight.data[5])

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_type_matrix, mask=None):
        B, N, D = node_features.shape
        H, D_h = self.num_heads, self.d_head

        Q = self.W_q(node_features).view(B, N, H, D_h).transpose(1, 2)
        K = self.W_k(node_features).view(B, N, H, D_h).transpose(1, 2)
        V = self.W_v(node_features).view(B, N, H, D_h).transpose(1, 2)

        k_bias_lookup = self.key_biases(edge_type_matrix)
        v_bias_lookup = self.value_biases(edge_type_matrix)

        K_plus_bias = K.unsqueeze(2) + k_bias_lookup.unsqueeze(1)
        attn_scores = (Q.unsqueeze(3) * K_plus_bias).sum(-1) * self.scale

        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask == 1, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        V_plus_bias = V.unsqueeze(2) + v_bias_lookup.unsqueeze(1)
        context = (attn_probs.unsqueeze(-1) * V_plus_bias).sum(dim=3)

        context = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(context)


class NumDAE(nn.Module):
    def __init__(self, args, num_ent, num_rel, dim_model, num_head, dim_hid, num_layer, dropout=0.1):
        super(NumDAE, self).__init__()
        self.args = args
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_layer = num_layer
        self.dropout = dropout

        self.struct_mode = 'positional'
        # self.struct_mode = 'gat'
        self.num_gat_layers = args.num_gat_layers if hasattr(args, 'num_gat_layers') else 1

        self.h_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.r_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.t_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.q_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.v_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))

        self.ent_embeddings = nn.Embedding(num_ent + 1 + num_rel, dim_model)
        self.rel_embeddings = nn.Embedding(num_rel + 1, dim_model)

        self.pri_enc = nn.Linear(dim_model * 3, dim_model)
        self.qv_enc = nn.Linear(dim_model * 2, dim_model)

        self.pri_enc_struct = nn.Linear(dim_model * 3, dim_model)
        self.qv_enc_struct = nn.Linear(dim_model * 2, dim_model)

        self.ent_dec = nn.Linear(dim_model, num_ent)
        self.rel_dec = nn.Linear(dim_model, num_rel)
        self.num_dec = nn.Linear(dim_model, num_rel)

        self.numerical_encoder = nn.Sequential(
            nn.Linear(1, dim_model // 2),
            nn.Tanh(),
            nn.Linear(dim_model // 2, dim_model)
        )

        self.emb_num_lit = Gate(dim_model * 2, dim_model)

        self.encoder = DiffTransformer(d_model=dim_model, num_heads=num_head, num_layers=num_layer,
                                       dropout_rate=dropout)
        # self.encoder = DiffEncoder(d_model=dim_model, num_heads=num_head, num_layers=num_layer,
        #                                dropout_rate=dropout)
        self.decoder = DiffTransformer(d_model=dim_model, num_heads=num_head, num_layers=num_layer,
                                       dropout_rate=dropout)
        # self.decoder = DiffDecoder(d_model=dim_model, num_heads=num_head, num_layers=num_layer,
        #                                dropout_rate=dropout)

        self.device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")

        self.learnable_param = nn.Parameter(torch.tensor(0.5))
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim_model * 2, dim_model),
            nn.Sigmoid()
        )

        if self.struct_mode == 'positional':
            self.pri_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
            self.qv_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        elif self.struct_mode == 'gat':
            self.num_edge_types = 6
            self.struct_attn_layers = nn.ModuleList([
                StructuralGraphAttention(dim_model, num_head, num_edge_types=self.num_edge_types, dropout=dropout)
                for _ in range(self.num_gat_layers)
            ])
            self.gat_layer_norms = nn.ModuleList([
                nn.LayerNorm(dim_model) for _ in range(self.num_gat_layers)
            ])
            self.gat_final_norm = nn.LayerNorm(dim_model)
            self.gat_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"unknown struct_mode: '{self.struct_mode}'")

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.h_pos)
        nn.init.xavier_uniform_(self.r_pos)
        nn.init.xavier_uniform_(self.t_pos)
        nn.init.xavier_uniform_(self.q_pos)
        nn.init.xavier_uniform_(self.v_pos)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.pri_enc.weight)
        nn.init.xavier_uniform_(self.pri_enc_struct.weight)
        nn.init.xavier_uniform_(self.qv_enc.weight)
        nn.init.xavier_uniform_(self.qv_enc_struct.weight)
        nn.init.xavier_uniform_(self.ent_dec.weight)
        nn.init.xavier_uniform_(self.rel_dec.weight)
        nn.init.xavier_uniform_(self.num_dec.weight)
        self.pri_enc.bias.data.zero_()
        self.pri_enc_struct.bias.data.zero_()
        self.qv_enc.bias.data.zero_()
        self.qv_enc_struct.bias.data.zero_()
        self.ent_dec.bias.data.zero_()
        self.rel_dec.bias.data.zero_()
        self.num_dec.bias.data.zero_()
        for layer in self.numerical_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()

        if self.struct_mode == 'positional':
            nn.init.xavier_uniform_(self.pri_pos)
            nn.init.xavier_uniform_(self.qv_pos)
        elif self.struct_mode == 'gat':
            for gat_layer in self.struct_attn_layers:
                nn.init.xavier_uniform_(gat_layer.W_q.weight)
                nn.init.xavier_uniform_(gat_layer.W_k.weight)
                nn.init.xavier_uniform_(gat_layer.W_v.weight)
                nn.init.xavier_uniform_(gat_layer.W_o.weight)
            for norm_layer in self.gat_layer_norms:
                norm_layer.weight.data.fill_(1.0)
                norm_layer.bias.data.zero_()
            self.gat_final_norm.weight.data.fill_(1.0)
            self.gat_final_norm.bias.data.zero_()

    def forward(self, src, num_values, src_key_padding_mask, mask_locs):
        batch_size = len(src)
        num_qualifiers = (src[..., 3::2].flatten().shape[0] // batch_size)
        N_nodes = 3 + 2 * num_qualifiers

        h_emb = self.ent_embeddings(src[..., 0]).view(batch_size, 1, self.dim_model)
        r_emb = self.rel_embeddings(src[..., 1]).view(batch_size, 1, self.dim_model)
        t_emb = self.ent_embeddings(src[..., 2]).view(batch_size, 1, self.dim_model)
        q_emb = self.rel_embeddings(src[..., 3::2].flatten()).view(batch_size, -1, self.dim_model)
        v_emb = self.ent_embeddings(src[..., 4::2].flatten()).view(batch_size, -1, self.dim_model)

        # --- Semantic Stream ---
        t_emb_sem = t_emb.clone()
        v_emb_sem = v_emb.clone()

        is_numeric_t_bool = (num_values[..., 0] != 1.0) & (num_values[..., 0] != -1.0)
        is_numeric_v_bool = (num_values[..., 1:] != 1.0) & (num_values[..., 1:] != -1.0)

        if is_numeric_t_bool.any():
            numeric_vals_t = num_values[is_numeric_t_bool, 0].unsqueeze(-1)
            numeric_emb_t_raw = self.numerical_encoder(numeric_vals_t)
            r_emb_context = r_emb[is_numeric_t_bool]
            numeric_emb_t_contextualized = numeric_emb_t_raw * r_emb_context.view(-1, self.dim_model)
            t_emb_sem[is_numeric_t_bool] = self.emb_num_lit(
                t_emb[is_numeric_t_bool].view(-1, self.dim_model),
                numeric_emb_t_contextualized
            ).view(-1, 1, self.dim_model)

        if is_numeric_v_bool.any():
            numeric_vals_v = num_values[..., 1:][is_numeric_v_bool].unsqueeze(-1)
            numeric_emb_v_raw = self.numerical_encoder(numeric_vals_v)
            q_emb_context = q_emb[is_numeric_v_bool]
            numeric_emb_v_contextualized = numeric_emb_v_raw * q_emb_context
            v_emb_sem[is_numeric_v_bool] = self.emb_num_lit(
                v_emb[is_numeric_v_bool].view(-1, self.dim_model),
                numeric_emb_v_contextualized
            )

        tri_seq_semantic = self.pri_enc(torch.cat([h_emb, r_emb, t_emb_sem], dim=-1))
        qv_seqs_semantic = self.qv_enc(torch.cat([q_emb, v_emb_sem], dim=-1))
        enc_in_seq_semantic = torch.cat([tri_seq_semantic, qv_seqs_semantic], dim=1)

        # --- Structural Stream ---
        if self.struct_mode == 'positional':
            tri_seq_struct = self.pri_enc_struct(torch.cat([h_emb, r_emb, t_emb], dim=-1))
            qv_seqs_struct = self.qv_enc_struct(torch.cat([q_emb, v_emb], dim=-1))
            enc_in_seq_struct_content = torch.cat([tri_seq_struct, qv_seqs_struct], dim=1)

            num_qualifiers_pos = qv_seqs_struct.shape[1]
            pri_final_pos_emb = self.pri_pos
            if num_qualifiers_pos > 0:
                qual_final_pos_embs = self.qv_pos.expand(1, num_qualifiers_pos, -1)
                final_pos_embs = torch.cat([pri_final_pos_emb, qual_final_pos_embs], dim=1)
            else:
                final_pos_embs = pri_final_pos_emb
            enc_in_seq_struct = enc_in_seq_struct_content + final_pos_embs

        elif self.struct_mode == 'gat':
            q_emb_gat = self.ent_embeddings(src[..., 3::2].flatten()).view(batch_size, -1, self.dim_model)
            graph_node_features = torch.cat([h_emb, r_emb, t_emb, q_emb_gat, v_emb], dim=1)

            is_numeric_node = torch.full((batch_size, N_nodes), False, dtype=torch.bool, device=self.device)
            is_numeric_node[:, 2] = is_numeric_t_bool
            if num_qualifiers > 0:
                v_indices = torch.arange(4, N_nodes, 2, device=self.device)
                is_numeric_node[:, v_indices] = is_numeric_v_bool
            edge_type_matrix = torch.full((batch_size, N_nodes, N_nodes), 5, dtype=torch.long, device=self.device)
            is_numeric_i = is_numeric_node.unsqueeze(2)
            is_numeric_j = is_numeric_node.unsqueeze(1)
            is_numeric_pair = is_numeric_i & is_numeric_j
            edge_type_matrix[is_numeric_pair] = 4
            edge_type_matrix[:, 0, 1] = 0
            edge_type_matrix[:, 1, 0] = 0
            edge_type_matrix[:, 1, 2] = 1
            edge_type_matrix[:, 2, 1] = 1
            if num_qualifiers > 0:
                q_indices = torch.arange(3, N_nodes, 2, device=self.device)
                edge_type_matrix[:, 1, q_indices] = 2
                edge_type_matrix[:, q_indices, 1] = 2
                v_indices = torch.arange(4, N_nodes, 2, device=self.device)
                batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1)
                edge_type_matrix[batch_indices, q_indices.unsqueeze(0), v_indices.unsqueeze(0)] = 3
                edge_type_matrix[batch_indices, v_indices.unsqueeze(0), q_indices.unsqueeze(0)] = 3
            edge_type_matrix.diagonal(dim1=-2, dim2=-1).fill_(5)
            pri_mask = torch.full((batch_size, 3), False, device=self.device)
            qual_mask = src_key_padding_mask[:, 1:]
            qv_mask = qual_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(batch_size, -1)
            gat_mask = torch.cat([pri_mask, qv_mask], dim=1)
            gat_output_nodes = graph_node_features
            for i in range(self.num_gat_layers):
                residual = gat_output_nodes
                nodes_norm = self.gat_layer_norms[i](gat_output_nodes)
                gat_out = self.struct_attn_layers[i](nodes_norm, edge_type_matrix, gat_mask)
                gat_output_nodes = residual + self.gat_dropout(gat_out)
            gat_output_nodes = self.gat_final_norm(gat_output_nodes)
            gat_h_emb = gat_output_nodes[:, 0:1, :]
            gat_r_emb = gat_output_nodes[:, 1:2, :]
            gat_t_emb = gat_output_nodes[:, 2:3, :]
            gat_q_emb = gat_output_nodes[:, 3:N_nodes:2, :]
            gat_v_emb = gat_output_nodes[:, 4:N_nodes:2, :]
            tri_seq_struct = self.pri_enc_struct(torch.cat([gat_h_emb, gat_r_emb, gat_t_emb], dim=-1))
            qv_seqs_struct = self.qv_enc_struct(torch.cat([gat_q_emb, gat_v_emb], dim=-1))
            enc_in_seq_struct = torch.cat([tri_seq_struct, qv_seqs_struct], dim=1)

        else:
            raise ValueError(f"unknown struct_mode: '{self.struct_mode}'.")

        # --- Differnential-Aware Embedding Transformer ---
        combined_features = torch.cat([enc_in_seq_struct, enc_in_seq_semantic], dim=-1)
        alpha = self.fusion_gate(combined_features)
        enc_in_fused = alpha * enc_in_seq_semantic + (1 - alpha) * enc_in_seq_struct
        enc_out_seq = self.encoder(enc_in_fused, mask=src_key_padding_mask)

        # --- Decoder ---
        dec_in_rep = enc_out_seq[mask_locs].view(batch_size, 1, self.dim_model)
        triplet = torch.stack([h_emb + self.h_pos, r_emb + self.r_pos, t_emb + self.t_pos], dim=2)
        qv = torch.stack([q_emb + self.q_pos, v_emb + self.v_pos, torch.zeros_like(v_emb)], dim=2)
        dec_in_part = torch.cat([triplet, qv], dim=1)[mask_locs]  # Shape: [Batch, 3, Dim]
        dec_query_seq = torch.cat([dec_in_rep, dec_in_part], dim=1)  # Shape: [Batch, 4, Dim]
        dec_in_seq = torch.cat([enc_out_seq, dec_query_seq], dim=1)  # Shape: [Batch, N+4, Dim]
        dec_suffix_mask = torch.full((batch_size, 4), False, device=self.device)
        non_zero_indices = torch.nonzero(mask_locs)
        if non_zero_indices.size(0) > 0:
            dec_suffix_mask[non_zero_indices[:, 0][non_zero_indices[:, 1] != 0], 3] = True
        dec_full_mask = torch.cat([src_key_padding_mask, dec_suffix_mask], dim=1)
        dec_out_full = self.decoder(dec_in_seq, mask=dec_full_mask)
        dec_out_seq = dec_out_full[:, -4:, :]


        return self.ent_dec(dec_out_seq), self.rel_dec(dec_out_seq), self.num_dec(dec_out_seq)