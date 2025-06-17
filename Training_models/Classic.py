import torch
import os
from lambeq import BobcatParser, PytorchModel, PytorchTrainer
from lambeq.backend.tensor import Dim
from lambeq import AtomicType, SpiderAnsatz
import matplotlib.pyplot as plt
import numpy as np

# Constants
BATCH_SIZE = 30
EPOCHS = 30
LEARNING_RATE = 3e-2
SEED = 0

def read_data(filename):
    """Reads data from a file and returns labels and sentences."""
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])  # First character is the label
            labels.append([t, 1 - t])  # Binary labels
            sentences.append(line[1:].strip())  # Sentence text
    return labels, sentences 

# Load datasets
train_labels, train_data = read_data('mc_train_data.txt')
val_labels, val_data = read_data('mc_dev_data.txt')
test_labels, test_data = read_data('mc_test_data.txt')

# Check if testing mode is enabled
TESTING = int(os.environ.get('TEST_NOTEBOOKS', '0'))
if TESTING:
    train_labels, train_data = train_labels[:2], train_data[:2]
    val_labels, val_data = val_labels[:2], val_data[:2]
    test_labels, test_data = test_labels[:2], test_data[:2]
    EPOCHS = 1  # Reduce epochs for testing

# Initialize the parser
parser = BobcatParser(verbose='text')

# Convert sentences to diagrams
train_diagrams = parser.sentences2diagrams(train_data)
val_diagrams = parser.sentences2diagrams(val_data)
test_diagrams = parser.sentences2diagrams(test_data)

# Define the ansatz for the model
ansatz = SpiderAnsatz({
    AtomicType.NOUN: Dim(2),
    AtomicType.SENTENCE: Dim(2)
})

# Create circuits from diagrams
train_circuits = [ansatz(diagram) for diagram in train_diagrams]
val_circuits = [ansatz(diagram) for diagram in val_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

# Visualize the first circuit
train_circuits[0].draw()

# Combine all circuits for model initialization
all_circuits = train_circuits + val_circuits + test_circuits
model = PytorchModel.from_diagrams(all_circuits)

# Define the sigmoid function
sig = torch.sigmoid

def accuracy(y_hat, y):
    """Calculates accuracy based on predictions and true labels."""
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y)) / len(y) / 2  # Half due to double-counting

# Evaluation metrics
eval_metrics = {"acc": accuracy}

# Initialize the trainer
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

# Create datasets for training and validation
from lambeq import Dataset

train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

# Train the model
trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=5)

# Plot training and validation metrics
fig1, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharey='row', figsize=(10, 6))
ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Epochs')
ax_br.set_xlabel('Epochs')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')

# Plotting colors
colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
range_ = np.arange(1, trainer.epochs + 1)

# Training loss and accuracy
ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))

# Validation loss and accuracy
ax_tr.plot(range_, trainer.val_costs, color=next(colours))
ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))

# Print test accuracy
test_acc = accuracy(model(test_circuits), torch.tensor(test_labels))
print('Test accuracy:', test_acc.item())

# Define a custom model
class MyCustomModel(PytorchModel):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 2)  # Simple linear layer

    def forward(self, input):
        """Define a custom forward pass."""
        preds = self.get_diagram_output(input)
        preds = self.net(preds.float())
        return preds

# Initialize and train the custom model
custom_model = MyCustomModel.from_diagrams(all_circuits)
custom_model_trainer = PytorchTrainer(
    model=custom_model,
    loss_function=torch.nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.AdamW,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    evaluate_functions=eval_metrics,
    evaluate_on_train=True,
    verbose='text',
    seed=SEED
)

# Train the custom model
custom_model_trainer.fit(train_dataset, val_dataset, log_interval=5)

# Plot training and validation metrics for the custom model
fig2, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharey='row', figsize=(10, 6))
ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Epochs')
ax_br.set_xlabel('Epochs')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')

# Plotting colors
colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
range_ = np.arange(1, custom_model_trainer.epochs + 1)

# Training loss and accuracy for custom model
ax_tl.plot(range_, custom_model_trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(range_, custom_model_trainer.train_eval_results['acc'], color=next(colours))

# Validation loss and accuracy for custom model
ax_tr.plot(range_, custom_model_trainer.val_costs, color=next(colours))
ax_br.plot(range_, custom_model_trainer.val_eval_results['acc'], color=next(colours))

# Print test accuracy for the custom model
test_acc_custom = accuracy(custom_model(test_circuits), torch.tensor(test_labels))
print('Test accuracy (custom model):', test_acc_custom.item())
