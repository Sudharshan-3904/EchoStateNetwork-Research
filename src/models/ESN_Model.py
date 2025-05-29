import matplotlib.pyplot as plt
import torch
import numpy as np
import openvino.runtime as ov
from openvino.runtime import Core
from datetime import datetime

class EchoStateNetworkModular:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, sparsity=0.1, input_scaling=1.0, leaking_rate=1.0):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate

        self.W_in = np.random.uniform(-self.input_scaling, self.input_scaling, (self.reservoir_size, self.input_size + 1))
        self.W = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.reservoir_size))
        
        mask = np.random.rand(*self.W.shape) > self.sparsity
        self.W[mask] = 0
        
        eigenvalues = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / eigenvalues

        self.W_out = None
        self.reservoir_state = np.zeros((self.reservoir_size, 1))

    def _update_reservoir(self, input_vector):
        input_vector = np.reshape(input_vector, (-1, 1))
        augmented_input = np.vstack((1, input_vector))
        pre_activation = np.dot(self.W_in, augmented_input) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + self.leaking_rate * np.tanh(pre_activation)

    def fit(self, inputs, targets, regularization=1e-8):
        states = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            states.append(self.reservoir_state.flatten())
        states = np.array(states)

        states = np.hstack((np.ones((states.shape[0], 1)), states))

        targets = np.array(targets)
        self.W_out = np.dot(np.linalg.pinv(np.dot(states.T, states) + regularization * np.eye(states.shape[1])), np.dot(states.T, targets))

    def predict(self, inputs):
        predictions = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            augmented_state = np.vstack((1, self.reservoir_state))
            output = np.dot(self.W_out.T, augmented_state)
            predictions.append(output.flatten())
        return np.array(predictions)


class EchoStateNetworkGFG:
    def __init__(self, reservoir_size, spectral_radius=0.9):
        self.reservoir_size = reservoir_size

        self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))

        self.W_in = np.random.rand(reservoir_size, 1) - 0.5

        self.W_out = None

    def train(self, input_data, target_data):
        reservoir_states = self.run_reservoir(input_data)

        self.W_out = np.dot(np.linalg.pinv(reservoir_states), target_data)

    def predict(self, input_data):
        reservoir_states = self.run_reservoir(input_data)

        predictions = np.dot(reservoir_states, self.W_out)

        return predictions

    def run_reservoir(self, input_data):
        reservoir_states = np.zeros((len(input_data), self.reservoir_size))

        for t in range(1, len(input_data)):
            reservoir_states[t, :] = np.tanh(
                np.dot(
                    self.W_res, reservoir_states[t - 1, :]) + np.dot(self.W_in, input_data[t])
            )

        return reservoir_states

class EchoStateNetworkGPU:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, sparsity=0.1, input_scaling=1.0, leaking_rate=1.0):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert weights to PyTorch tensors on GPU
        self.W_in = torch.tensor(
            np.random.uniform(-self.input_scaling, self.input_scaling, (self.reservoir_size, self.input_size + 1)),
            device=self.device, dtype=torch.float32
        )
        self.W = torch.tensor(
            np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.reservoir_size)),
            device=self.device, dtype=torch.float32
        )
        
        mask = torch.tensor(np.random.rand(*self.W.shape) > self.sparsity, device=self.device)
        self.W = self.W * mask
        
        eigenvalues = torch.linalg.eigvals(self.W)
        self.W *= self.spectral_radius / torch.max(torch.abs(eigenvalues))

        self.W_out = None
        self.reservoir_state = torch.zeros((self.reservoir_size, 1), device=self.device)

    def reset_state(self):
        """Reset the reservoir state"""
        self.reservoir_state = torch.zeros((self.reservoir_size, 1), device=self.device)

    def save_state(self):
        """Save current reservoir state"""
        return self.reservoir_state.clone()

    def load_state(self, state):
        """Load a previously saved reservoir state"""
        self.reservoir_state = state.clone()

    def _update_reservoir(self, input_vector):
        # Convert input to PyTorch tensor if it's not already
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.tensor(input_vector, device=self.device, dtype=torch.float32)
        input_vector = input_vector.reshape(-1, 1)
        augmented_input = torch.vstack((torch.ones(1, device=self.device, dtype=torch.float32), input_vector))
        pre_activation = torch.matmul(self.W_in, augmented_input) + torch.matmul(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + self.leaking_rate * torch.tanh(pre_activation)

    def train(self, inputs, targets, ridge_param=1e-6):
        """Train the ESN using ridge regression"""
        states = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            states.append(torch.vstack((
                torch.ones(1, device=self.device),
                self.reservoir_state
            )).flatten())
        
        # Collect states into matrix
        states_matrix = torch.stack(states)
        
        # Move targets to GPU if needed
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, device=self.device, dtype=torch.float32)
        
        # Ridge regression on GPU
        identity = torch.eye(states_matrix.shape[1], device=self.device)
        A = torch.matmul(states_matrix.T, states_matrix) + ridge_param * identity
        B = torch.matmul(states_matrix.T, targets)
        self.W_out = torch.linalg.solve(A, B)
        
        self.reset_state()
        return self

    def predict(self, inputs):
        predictions = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            augmented_state = torch.vstack((
                torch.ones(1, device=self.device, dtype=torch.float32),
                self.reservoir_state
            ))
            # Ensure W_out is float32
            if self.W_out is not None and self.W_out.dtype != torch.float32:
                self.W_out = self.W_out.to(dtype=torch.float32)
            output = torch.matmul(self.W_out.T, augmented_state)
            predictions.append(output.cpu().numpy().flatten())
        return np.array(predictions)

class ESNModelGPU:
    def __init__(self, cpu_model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            tprint(f"Using GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            tprint(f"GPU Memory: {total_memory:.2f} GB")
        else:
            tprint("CUDA not available. Using CPU.")
        
        self.input_size = cpu_model.input_size
        self.reservoir_size = cpu_model.reservoir_size
        self.output_size = cpu_model.output_size
        self.spectral_radius = cpu_model.spectral_radius
        self.sparsity = cpu_model.sparsity
        self.input_scaling = cpu_model.input_scaling
        self.leaking_rate = cpu_model.leaking_rate
        
        self.W_in = torch.tensor(cpu_model.W_in, device=self.device, dtype=torch.float32)
        self.W = torch.tensor(cpu_model.W, device=self.device, dtype=torch.float32)
        self.W_out = torch.tensor(cpu_model.W_out, device=self.device, dtype=torch.float32) if cpu_model.W_out is not None else None
        self.reservoir_state = torch.tensor(cpu_model.reservoir_state, device=self.device, dtype=torch.float32)
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def _update_reservoir(self, input_vector):
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, device=self.device, dtype=torch.float32)
        
        input_with_bias = torch.cat([torch.ones(1, device=self.device), input_vector])
        new_state = torch.tanh(torch.matmul(self.W_in, input_with_bias) + 
                              torch.matmul(self.W, self.reservoir_state))
        
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + self.leaking_rate * new_state
    
    def predict(self, inputs):
        """Predict outputs for the given inputs."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        
        predictions = []
        with torch.no_grad():
            for i in range(inputs.shape[0]):
                input_vector = inputs[i]
                self._update_reservoir(input_vector)
                augmented_state = torch.cat([torch.ones(1, device=self.device), self.reservoir_state.flatten()])
                output = torch.matmul(self.W_out.T, augmented_state)
                predictions.append(output.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_with_temperature(self, inputs, temperature=0.7):
        """Predict with temperature scaling for more diverse outputs."""
        predictions = self.predict(inputs)
        scaled_predictions = predictions / temperature
        
        return scaled_predictions

class EnhancedESN(EchoStateNetworkModular):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.99, 
                 sparsity=0.05, input_scaling=1.2, leaking_rate=0.2):
        super().__init__(input_size, reservoir_size, output_size, 
                         spectral_radius, sparsity, input_scaling, leaking_rate)
        
    def fit(self, inputs, targets, regularization=1e-9):
        """Enhanced fit method with lower regularization"""
        states = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            states.append(self.reservoir_state.flatten())
        states = np.array(states)

        states = np.hstack((np.ones((states.shape[0], 1)), states))

        targets = np.array(targets)
        self.W_out = np.dot(np.linalg.pinv(np.dot(states.T, states) + regularization * np.eye(states.shape[1])), np.dot(states.T, targets))
        
    def predict_with_temperature(self, inputs, temperature=0.7):
        """Predict with temperature scaling for more diverse outputs"""
        predictions = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            augmented_state = np.vstack((1, self.reservoir_state))
            output = np.dot(self.W_out.T, augmented_state)
            
            # Apply temperature scaling
            output = output / temperature
            predictions.append(output.flatten())
        return np.array(predictions)
    

class EchoStateNetworkNPU:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, 
                 sparsity=0.1, input_scaling=1.0, leaking_rate=1.0):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate

        # Initialize weights
        self.W_in = np.random.uniform(-self.input_scaling, self.input_scaling, 
                                    (self.reservoir_size, self.input_size + 1))
        self.W = np.random.uniform(-0.5, 0.5, 
                                 (self.reservoir_size, self.reservoir_size))
        
        # Apply sparsity
        mask = (np.random.rand(*self.W.shape) > self.sparsity)
        self.W *= mask
        
        # Scale by spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        self.W *= self.spectral_radius / max_abs_eigenvalue

        self.W_out = None
        self.reservoir_state = np.zeros((self.reservoir_size, 1))

        # Initialize OpenVINO Core and check for NPU
        self.core = Core()
        self.devices = self.core.available_devices
        if "NPU" not in self.devices:
            tprint("Warning: NPU not found, available devices:", self.devices)
            self.device = "CPU"  # Fallback to CPU
        else:
            self.device = "NPU"
            try:
                device_info = self.core.get_property("NPU", "FULL_DEVICE_NAME")
                tprint(f"Using Intel NPU: {device_info}")
            except:
                tprint("Using Intel NPU (details unavailable)")

        # Set NPU configuration
        self.config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "FP16",
            "NUM_STREAMS": "1",
            "PERF_COUNT": "NO"
        }
        
        self.compile_model()

    def compile_model(self):
        try:
            # Create OpenVINO model with optimized configuration
            self.update_model = self.core.compile_model(
                self._create_reservoir_update_model(), 
                device_name=self.device,
                config=self.config
            )
            self.update_infer = self.update_model.create_infer_request()
            tprint(f"Model compiled successfully for {self.device}")
        except Exception as e:
            tprint(f"Error compiling model for {self.device}: {str(e)}")
            tprint("Falling back to CPU...")
            self.device = "GPU"
            # Try again with GPU
            try:
                self.update_model = self.core.compile_model(
                    self._create_reservoir_update_model(), 
                    device_name=self.device
                )
                self.update_infer = self.update_model.create_infer_request()
                tprint("Model compiled successfully on CPU")
            except Exception as e2:
                tprint(f"Fatal error: Could not compile model on CPU: {str(e2)}")
                raise

    def _create_reservoir_update_model(self):
        # Create OpenVINO model for reservoir update computation
        # Fix: Use correct OpenVINO data type definition
        input_node = ov.opset8.parameter(
            [1, self.input_size], 
            dtype=ov.Type.f32
        )
        state_node = ov.opset8.parameter(
            [self.reservoir_size, 1], 
            dtype=ov.Type.f32
        )
        
        # Convert weights to constants with correct data type
        w_in_node = ov.opset8.constant(
            self.W_in.astype(np.float32), 
            dtype=ov.Type.f32
        )
        w_node = ov.opset8.constant(
            self.W.astype(np.float32), 
            dtype=ov.Type.f32
        )
        
        # Create computation graph
        new_state = ov.opset8.tanh(
            ov.opset8.add(
                ov.opset8.matmul(w_in_node, input_node),
                ov.opset8.matmul(w_node, state_node)
            )
        )
        
        return ov.Model([input_node, state_node], [new_state], "reservoir_update")

    def _update_reservoir(self, input_vector):
        # Update reservoir state using compiled OpenVINO model
        input_tensor = np.array(input_vector).reshape(1, -1)
        self.update_infer.set_input_tensors([input_tensor, self.reservoir_state])
        self.update_infer.infer()
        self.reservoir_state = self.update_infer.get_output_tensor().data

    def fit(self, inputs, targets, regularization=1e-8):
        states = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            states.append(self.reservoir_state.flatten())
        states = np.array(states)

        # Add bias term
        states = np.hstack((np.ones((states.shape[0], 1)), states))
        
        # Compute output weights using Ridge Regression
        targets = np.array(targets)
        identity = np.eye(states.shape[1])
        temp = np.dot(states.T, states) + regularization * identity
        self.W_out = np.dot(np.linalg.pinv(temp), np.dot(states.T, targets))

    def predict(self, inputs):
        predictions = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            augmented_state = np.vstack((1, self.reservoir_state))
            output = np.dot(self.W_out.T, augmented_state)
            predictions.append(output.flatten())
        return np.array(predictions)

def tprint(*args, **kwargs):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}]", *args, **kwargs)
