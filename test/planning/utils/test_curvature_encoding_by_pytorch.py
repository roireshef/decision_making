import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from decision_making.test.planning.utils.sample_curvature import sample_curvature


def test_curvature_encoding_by_pytorch():

    train_dataset = create_tensor_dataset(10000)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=64, shuffle=True)

    net = Net(train_dataset.data_tensor.size(1))
    pytorch_train(net, train_loader)

    test_dataset = create_tensor_dataset(5000)
    test_loader = data_utils.DataLoader(test_dataset)
    pytorch_test(net, test_loader)


def create_tensor_dataset(num_samples: int) -> TensorDataset:
    features, targets = sample_curvature(num_samples)
    torch_features = torch.from_numpy(features).float()
    torch_targets = torch.from_numpy(targets).float()
    dataset = data_utils.TensorDataset(torch_features, torch_targets)
    return dataset


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        # x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def pytorch_train(net: Net, trainloader: DataLoader):

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        steps = 0

        # With a batch size of 64 in each iteration
        for i, data in enumerate(trainloader, 0):  # trainloader reads data
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels.float())

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            steps += 1

        print(f"Train loss {running_loss/steps}")

    print('Finished Training')


def pytorch_test(net: Net, testloader: DataLoader):
    total_error = 0.
    samples = 0.
    classification_errors = 0
    output_thresh = 3
    for data in testloader:
        inputs, labels = data
        outputs = net(Variable(inputs))
        loss = outputs.data[0, 0] - labels[0]
        total_error += loss ** 2
        classification_errors += int((outputs.data[0, 0] - output_thresh) * (labels[0] - output_thresh) < 0)
        samples += 1
        print('Test loss %f' % loss)
    print('test error = %f; classification error=%f' % (np.sqrt(total_error/samples), classification_errors/samples))


# def pytorch_train1(features, targets):
#
#     # N is batch size; D_in is input dimension;
#     # H is hidden dimension; D_out is output dimension.
#     batch_size = 64
#     D_in, H, D_out = features.size(1), 100, 1
#
#     # Use the nn package to define our model and loss function.
#     model = torch.nn.Sequential(
#         torch.nn.Linear(D_in, H),
#         torch.nn.ReLU(),
#         torch.nn.Linear(H, D_out),
#     )
#     loss_fn = torch.nn.MSELoss(size_average=False)
#
#     # Use the optim package to define an Optimizer that will update the weights of
#     # the model for us. Here we will use Adam; the optim package contains many other
#     # optimization algoriths. The first argument to the Adam constructor tells the
#     # optimizer which Tensors it should update.
#     learning_rate = 1e-4
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     for epoch in range(1000):  # loop over the dataset multiple times
#
#         idxs = np.random.choice(features.size(0), batch_size)
#         x = features[torch.LongTensor(idxs)]
#         y = targets[torch.LongTensor(idxs)]
#
#         for t in range(500):
#             # Forward pass: compute predicted y by passing x to the model.
#             y_pred = model(x)
#
#             # Compute and print loss.
#             loss = loss_fn(y_pred, y)
#             print(t, loss.item())
#
#             # Before the backward pass, use the optimizer object to zero all of the
#             # gradients for the Tensors it will update (which are the learnable weights
#             # of the model)
#             optimizer.zero_grad()
#
#             # Backward pass: compute gradient of the loss with respect to model parameters
#             loss.backward()
#
#             # Calling the step function on an Optimizer makes an update to its parameters
#             optimizer.step()
