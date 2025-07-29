import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot_notebook_pipeline.dataset_utils.training import train_model

class MockPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch):
        # A simple forward pass that returns a dummy loss
        return torch.tensor(0.5, requires_grad=True), None

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"observation": torch.randn(1), "action": torch.randn(1)}

@pytest.fixture
def mock_policy():
    """A pytest fixture to provide a mock policy for testing."""
    return MockPolicy()

@pytest.fixture
def mock_dataloader():
    """A pytest fixture to provide a mock dataloader for testing."""
    dataset = MockDataset()
    return DataLoader(dataset, batch_size=2)

def test_train_model(mock_policy, mock_dataloader):
    """
    Tests the train_model function.
    """
    optimizer = torch.optim.Adam(mock_policy.parameters(), lr=1e-4)
    device = torch.device("cpu")
    
    try:
        train_model(mock_policy, mock_dataloader, optimizer, training_steps=5, log_freq=2, device=device)
    except Exception as e:
        pytest.fail(f"train_model raised an exception: {e}") 