import torch.nn as nn

from maps import fnn, attention, Conv1D
from config import args

class three_bus_PN(nn.Module):
    def __init__(
        self, 
        dynamic,
        algebraic,
        dyn_in_transform=None,
        dyn_out_transform=None,
        alg_in_transform=None,
        alg_out_transform=None,
        stacked=True,
        ):
        super(three_bus_PN, self).__init__()
        self.stacked = stacked
        self.dim = args.state_dim
        self.alg_dim = args.alg_state_dim + args.input_state_dim
        self.num_IRK_stages = dynamic.num_IRK_stages
        if dynamic.type == "attention":
            if stacked:
                self.Y = nn.ModuleList([
                    attention(
                        dynamic.layer_size,
                        dynamic.activation,
                        dynamic.initializer,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
                    for _ in range(self.dim)])
            else:
                self.Y = attention(
                        dynamic.layer_size,
                        dynamic.activation,
                        dynamic.initializer,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
        else:
            raise ValueError("{} type on NN not implemented".format(dynamic.type))


        if algebraic.type == "attention":
            if stacked:
                self.Z = nn.ModuleList([
                    attention(
                        algebraic.layer_size, 
                        algebraic.activation,
                        algebraic.initializer,
                        dropout_rate=algebraic.dropout_rate,
                        batch_normalization=algebraic.batch_normalization, 
                        layer_normalization=algebraic.layer_normalization, 
                        input_transform=alg_in_transform, 
                        output_transform=alg_out_transform,
                    )
                    for _ in range(self.alg_dim)])
            else:
                self.Z = attention(
                    algebraic.layer_size, 
                    algebraic.activation,
                    algebraic.initializer,
                    dropout_rate=algebraic.dropout_rate,
                    batch_normalization=algebraic.batch_normalization, 
                    layer_normalization=algebraic.layer_normalization, 
                    input_transform=alg_in_transform, 
                    output_transform=alg_out_transform,
                    )
        else:
            raise ValueError("{} type on NN not implemented".format(algebraic.type))

    def forward(self, input):
        if self.stacked:
            Y_parts = [module(input) for module in self.Y]
        else:
            dim_out = self.num_IRK_stages + 1
            Y = self.Y(input)
            Y_parts = [Y[..., i*dim_out : (i+1)*dim_out] for i in range(12)]

        Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11 = Y_parts
        
        if self.stacked:
            Z_parts = [module(input) for module in self.Z]
        else:
            dim_out = self.num_IRK_stages + 1
            Z = self.Z(input)  # Y ha dimensione (..., 12 * dim_out)
            Z_parts = [Z[..., i*dim_out : (i+1)*dim_out] for i in range(11)]

        Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10 = Z_parts
        
        return Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10