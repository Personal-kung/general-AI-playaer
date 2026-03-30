import torch.optim as optim

def train(model, buffer, batch_size=64, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.train()

    for epoch in range(epochs):
        batch = buffer.sample(batch_size)
        # Prepare tensors
        states, target_pis, target_vs = zip(*batch)
        
        # AlphaZero inputs are [Batch, Channels, Rows, Cols]
        s_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1) 
        pi_tensor = torch.FloatTensor(np.array(target_pis))
        v_tensor = torch.FloatTensor(np.array(target_vs)).unsqueeze(1)

        # Forward pass
        out_pi, out_v = model(s_tensor)

        # Loss = (v - target_v)^2 - pi * log(out_pi)
        value_loss = F.mse_loss(out_v, v_tensor)
        policy_loss = -torch.sum(pi_tensor * out_pi) / pi_tensor.size(0)
        total_loss = value_loss + policy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    print(f"Training Complete. Loss: {total_loss.item():.4f}")