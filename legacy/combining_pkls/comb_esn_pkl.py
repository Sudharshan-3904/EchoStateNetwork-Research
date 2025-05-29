import pickle
from src.models.ESN_Model import EchoStateNetworkGPU

file1_path = 'models/esn_model_gpu_batch_1.pkl'
file2_path = 'models/esn_model_gpu_batch_2.pkl'

output_file_path = 'SavedModels/combined_model.pkl'

with open(file1_path, 'rb') as file1:
    esn1 = pickle.load(file1)

with open(file2_path, 'rb') as file2:
    esn2 = pickle.load(file2)

combined_W_in = esn1.W_in + esn2.W_in                                   # W_in is a tensor
combined_W = esn1.W + esn2.W                                            # W is a tensor
combined_W_out = None
if esn1.W_out is not None and esn2.W_out is not None:
    combined_W_out = esn1.W_out + esn2.W_out                            # W_out is a tensor
combined_reservoir_state = esn1.reservoir_state + esn2.reservoir_state  # reservoir_state is a tensor

combined_esn = EchoStateNetworkGPU(
    input_size=esn1.input_size,
    reservoir_size=esn1.reservoir_size,
    output_size=esn1.output_size,
    spectral_radius=esn1.spectral_radius,
    sparsity=esn1.sparsity,
    input_scaling=esn1.input_scaling,
    leaking_rate=esn1.leaking_rate,
    device=esn1.device
)
combined_esn.W_in = combined_W_in
combined_esn.W = combined_W
combined_esn.W_out = combined_W_out
combined_esn.reservoir_state = combined_reservoir_state

with open(output_file_path, 'wb') as output_file:
    pickle.dump(combined_esn, output_file)

print(f"Combined .pkl file saved to {output_file_path}")