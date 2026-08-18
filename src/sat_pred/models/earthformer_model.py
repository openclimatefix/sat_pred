from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel


class Earthformer(CuboidTransformerModel):
    
    def forward(self, X, verbose=False):
        # The cloudcasting dataloader created batches of shape: 
        # (batch, channel, time, height, width)
        # Earthformer expects shape: (batch, time, height, width, channel)
        X = X.permute(0, 2, 3, 4, 1).contiguous()
        y_hat = super().forward(X, verbose=verbose)

        # Transpose back to cloudcasting shape
        return y_hat.permute(0, 4, 1, 2, 3)