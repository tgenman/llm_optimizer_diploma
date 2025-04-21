import torch

# Definition of a single linear attention unit for linear-regression data
# P is the value matrix
# Q is the product of key,query matrices
# the dimensions of the input are
# B: batch-size of prompts
# N: context length (excluding query)
# d: covariate dimension
# P,Q are d x d matrices
# Z is a B x (N+1) + (d+1) matrix
# Output is also B x (N+1) + (d+1)

# For linear attention, activation = None
# For standard attention, activation(x) = torch.nn.functional.softmax(x, dim = 2)
# For ReLU attention, activation(x) = torch.nn.relu(x)

def attention(P, Q, Z, activation=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = Z.shape[0]
    N = Z.shape[1] - 1
    d = Z.shape[2] - 1

    # Expand P and Q into (d+1) x (d+1)
    P_full = torch.cat([P, torch.zeros(1, d).to(device)], dim=0)
    P_full = torch.cat([P_full, torch.zeros(d + 1, 1).to(device)], dim=1)
    P_full[d, d] = 1  # extra dimension for the label

    Q_full = torch.cat([Q, torch.zeros(1, d).to(device)], dim=0)
    Q_full = torch.cat([Q_full, torch.zeros(d + 1, 1).to(device)], dim=1)

    # A is eye(N+1) except the bottom-right corner is 0
    A = torch.eye(N + 1).to(device)
    A[N, N] = 0

    # Attn shape: [B, N+1, N+1]
    Attn = torch.einsum('BNi, ij, BMj -> BNM', (Z, Q_full, Z))
    if activation is not None:
        Attn = activation(Attn)

    # key shape: [B, N+1, d+1]
    key = torch.einsum('ij, BNj -> BNi', (P_full, Z))

    # Output shape: [B, N+1, d+1]
    Output = torch.einsum('BNM,ML, BLi -> BNi', (Attn, A, key))
    return Output / N 