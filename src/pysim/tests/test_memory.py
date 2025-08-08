import pytest
torch = pytest.importorskip("torch")
from src.pysim.memory import Memory


def test_create_state_tuple_no_padding():
    memory = Memory()

    samples = []
    for i in range(3):
        f0 = torch.full((2, 2), float(i))
        f1 = torch.tensor([i, i + 1, i + 2], dtype=torch.float32)
        samples.append((f0, f1))

    state_fields = list(zip(*samples))
    (field0, field1), mask = memory.create_state_tuple(state_fields)

    expected0 = torch.stack([s[0] for s in samples]).to(memory.device)
    expected1 = torch.stack([s[1] for s in samples]).to(memory.device)

    assert torch.equal(field0, expected0)
    assert torch.equal(field1, expected1)
    assert mask.numel() == 0
    assert mask.shape == torch.Size([0])


def test_create_state_tuple_with_padding():
    memory = Memory()
    device = memory.device
    samples = [
        (torch.ones((2, 2, 3)), torch.tensor([1.0])),
        (torch.ones((2, 3, 3)) * 2, torch.tensor([2.0])),
        (torch.ones((2, 1, 3)) * 3, torch.tensor([3.0])),
    ]

    state_fields = list(zip(*samples))
    (field0, field1), mask = memory.create_state_tuple(state_fields)

    assert field0.shape == (3, 2, 3, 3)
    assert field1.shape == (3, 1)
    assert mask.shape == (3, 3)

    lengths = [2, 3, 1]
    max_len = max(lengths)
    for i, L in enumerate(lengths):
        # valid data regions are unmasked
        assert torch.all(mask[i, :L] == False)
        # padded regions are masked and zeroed
        if L < max_len:
            assert torch.all(mask[i, L:])
            assert torch.all(field0[i, :, L:, :] == 0)
        assert torch.allclose(field0[i, :, :L, :], samples[i][0].to(device))

    expected1 = torch.stack([s[1] for s in samples]).to(device)
    assert torch.allclose(field1, expected1)

if __name__ == "__main__":
    pytest.main([__file__])