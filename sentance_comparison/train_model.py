import torch
import os
from lambeq import BobcatParser, PytorchModel, PytorchTrainer, Dataset
from lambeq.backend.tensor import Dim
from lambeq import AtomicType, CircuitAnsatz
from lambeq.backend.quantum import CX, Id, Ry

# Constants
BATCH_SIZE = 30
EPOCHS = 30
LEARNING_RATE = 3e-2
SEED = 0
SAVE_PATH = 'trained_pytorch_model.pt'

def read_data(filename):
    """Reads data from a file and returns labels and sentences."""
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])  # First character is the label
            labels.append([t, 1 - t])  # Binary labels
            sentences.append(line[1:].strip())  # Sentence text
    return labels, sentences 

class RealAnsatz(CircuitAnsatz):
    def __init__(self, ob_map, n_layers, n_single_qubit_params=1, discard=False):
        super().__init__(ob_map, n_layers, n_single_qubit_params, discard, [Ry, ])

    def params_shape(self, n_qubits):
        return (self.n_layers + 1, n_qubits)
    
    def circuit(self, n_qubits, params):
        circuit = Id(n_qubits)
        n_layers = params.shape[0] - 1

        for i in range(n_layers):
            syms = params[i]

            # adds a layer of Y rotations
            circuit >>= Id().tensor(*[Ry(sym) for sym in syms])

            # adds a ladder of CNOTs
            for j in range(n_qubits - 1):
                circuit >>= Id(j) @ CX @ Id(n_qubits - j - 2)

        # adds a final layer of Y rotations
        circuit >>= Id().tensor(*[Ry(sym) for sym in params[-1]])

        return circuit

def main():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    # Load data
    train_labels, train_data = read_data('mc_train_data.txt')
    val_labels, val_data = read_data('mc_dev_data.txt')

    # Initialize parser
    parser = BobcatParser(verbose='text')

    # Define the number of qubits and layers for the RealAnsatz
    n_qubits = 2  # Adjust based on your problem
    n_layers = 3  # Number of layers in the ansatz

    # Initialize RealAnsatz
    ansatz = RealAnsatz(
        ob_map={AtomicType.NOUN: Dim(n_qubits), AtomicType.SENTENCE: Dim(n_qubits)},
        n_layers=n_layers
    )

    # Parse sentences to diagrams
    train_diagrams = parser.sentences2diagrams(train_data)
    val_diagrams = parser.sentences2diagrams(val_data)

    # Convert diagrams to circuits
    train_circuits = [ansatz(diagram) for diagram in train_diagrams]
    val_circuits = [ansatz(diagram) for diagram in val_diagrams]

    # Combine all circuits (train + val) to initialize model parameters
    all_circuits = train_circuits + val_circuits

    # Initialize model from diagrams
    model = PytorchModel.from_diagrams(all_circuits)

    # Define accuracy metric
    sig = torch.sigmoid
    def accuracy(y_hat, y):
        return torch.sum(torch.eq(torch.round(sig(y_hat)), y)) / len(y) / 2  # half due to double-counting
    eval_metrics = {"acc": accuracy}

    # Create Dataset objects
    train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
    val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

    # Initialize trainer
    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED
    )

    # Train the model
    trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=5)

    # Save the trained model parameters and circuit structure
    torch.save({
        'model_state_dict': model.state_dict(),
        'circuits': all_circuits
    }, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == '__main__':
    main()
