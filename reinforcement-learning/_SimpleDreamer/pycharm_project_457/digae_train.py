from DiGAE.layers import DirectedGCNConvEncoder, DirectedInnerProductDecoder
from DiGAE.models import DirectedGAE

alpha = 1.0
beta = 0.0
self_loops = True
adaptive = False
hidden_channels = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = DirectedGCNConvEncoder(
    in_channels, 
    hidden_channels,
    out_channels, 
    alpha=alpha,
    beta=beta,
    self_loops=self_loops,
    adaptive=adaptive
)
decoder = DirectedInnerProductDecoder()
model = DirectedGAE(encoder, decoder)
model = model.to(device)
