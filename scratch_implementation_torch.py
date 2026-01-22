import torch

N = 10
D_in = 1
D_out = 1

X = torch.randn(N, D_in)
true_W = torch.tensor([[2.0]])
true_b = torch.tensor([1.0])
Y = X @ true_W + true_b # + 0.1 * torch.randn(N, D_out) (noise)

learning_rate, epochs = 0.01, 1001

W, b = torch.randn(1, 1, requires_grad=True), torch.randn(1, requires_grad=True)

for epoch in range(epochs):
    y_hat = X @ W + b
    loss = torch.mean((Y - y_hat) ** 2)

    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss = {loss.item()}, W = {W.item():.3f}, b = {b.item():.3f}')

    W.grad.zero_()
    b.grad.zero_()

# y = a * x

print(f'Final parameters: W = {W.item():.3f}, b = {b.item():.3f}')
print(f'True parameters: W = {true_W.item():.3f}, b = {true_b.item():.3f}')