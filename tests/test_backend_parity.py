import numpy as np
import pytest
try:
    import torch
except ImportError:
    torch = None

from recon_lite.learning.baseline import BaselineLearner, ComputeBackend, apply_sensor, compute_sensor_xp

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_backend_parity_sensor_apply():
    feature_dim = 10
    batch_size = 5
    v_batch = np.random.rand(batch_size, feature_dim).astype(np.float32)
    
    # Create learner with numpy backend
    learner_np = BaselineLearner(feature_dim=feature_dim, device="numpy")
    sensor = learner_np.spawn_sensor()
    learner_np.sensors.append(sensor)
    
    # Create learner with torch backend
    learner_torch = BaselineLearner(feature_dim=feature_dim, device="cpu")
    # Share the same sensor (id, mask, etc)
    learner_torch.sensors.append(sensor)
    
    # Apply via numpy
    output_np = learner_np.batch_apply_sensors(v_batch)[sensor.id]
    
    # Apply via torch
    output_torch = learner_torch.batch_apply_sensors(v_batch)[sensor.id]
    
    # Check parity
    if isinstance(output_torch, torch.Tensor):
        output_torch = output_torch.cpu().numpy()
        
    np.testing.assert_allclose(output_np, output_torch, atol=1e-6)

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_backend_parity_xp_computation():
    feature_dim = 10
    num_pos = 20
    num_neg = 20
    
    delta_pos = np.random.rand(num_pos).astype(np.float32)
    delta_neg = np.random.rand(num_neg).astype(np.float32)
    
    learner_np = BaselineLearner(feature_dim=feature_dim, device="numpy")
    learner_torch = BaselineLearner(feature_dim=feature_dim, device="cpu")
    
    sensor = learner_np.spawn_sensor()
    
    xp_np = compute_sensor_xp(sensor, delta_pos, delta_neg, backend=learner_np.backend)
    xp_torch = compute_sensor_xp(sensor, delta_pos, delta_neg, backend=learner_torch.backend)
    
    assert pytest.approx(xp_np, abs=1e-6) == xp_torch

if __name__ == "__main__":
    # Manual run if pytest not used
    if torch:
        test_backend_parity_sensor_apply()
        test_backend_parity_xp_computation()
        print("Parity tests passed!")
    else:
        print("Skipping parity tests (torch not installed)")
