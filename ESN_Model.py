import matplotlib.pyplot as plt
import numpy as np

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

def sine_prediction_example_gfg(reservoir_size=100):
    time = np.arange(0, 20, 0.1)
    noise = 0.1 * np.random.rand(len(time))
    sine_wave_target = np.sin(time)
    # reservoir_size = 1000

    esn = EchoStateNetworkGFG(reservoir_size)

    training_input = noise[:, None]
    training_target = sine_wave_target[:, None]

    esn.train(training_input, training_target)

    test_input = noise[:, None]

    training_input = noise[:, None]
    training_target = sine_wave_target[:, None]

    esn.train(training_input, training_target)

    test_input = noise[:, None]

    predictions = esn.predict(test_input)

    plt.figure(figsize=(10, 6))
    plt.plot(time, sine_wave_target, label='True Sine Wave',
            linestyle='--', marker='o')
    plt.plot(time, predictions, label='ESN Prediction', linestyle='--', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Echo State Network Learning to Generate Sine Wave')
    plt.show()

sine_prediction_example_gfg()