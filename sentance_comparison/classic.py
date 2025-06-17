import torch
from lambeq import BobcatParser, PytorchModel
from lambeq.backend.tensor import Dim
from lambeq import AtomicType, SpiderAnsatz
import torch.nn.functional as F

def get_sentence_embedding(sentence, parser, ansatz, model, device):
    """
    Convert a sentence to its embedding using the model's diagram output.
    Args:
        sentence (str): Input sentence.
        parser (BobcatParser): Sentence parser.
        ansatz (SpiderAnsatz): Ansatze used to generate circuits.
        model (PytorchModel): Trained model.
        device (torch.device): Device (cpu or cuda).
    Returns:
        torch.Tensor: Embedding vector for the sentence.
    """
    # Parse sentence to diagram
    diagram = parser.sentence2diagram(sentence)
    # Generate circuit from diagram
    circuit = ansatz(diagram)
    # Move model and inputs to device
    model = model.to(device)
    # Prepare circuit batch with one element
    circuits = [circuit]
    # Get diagram outputs (embeddings) from model: batch_size x embedding_dim tensor
    with torch.no_grad():
        embeddings = model.get_diagram_output(circuits)
    # embeddings shape: (batch_size, dim), take first element
    embedding = embeddings[0].to(device)
    return embedding

def cosine_similarity(sent1, sent2, parser, ansatz, model, device):
    """
    Compute cosine similarity between embeddings of two sentences.
    
    Returns a value between -1 and 1.
    """
    emb1 = get_sentence_embedding(sent1, parser, ansatz, model, device)
    emb2 = get_sentence_embedding(sent2, parser, ansatz, model, device)
    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    return sim.item()  # scalar float

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading parser and ansatz...")
    parser = BobcatParser(verbose='text')
    ansatz = SpiderAnsatz({
        AtomicType.NOUN: Dim(2),
        AtomicType.SENTENCE: Dim(2)
    })

    # Since you might not have pretrained weights,
    # create a model from empty diagrams (dummy) to instantiate PytorchModel
    # If you have a pre-trained model, you'd load its weights here.
    dummy_diagram = parser.sentence2diagram("This is a dummy sentence.")
    dummy_circuit = ansatz(dummy_diagram)
    model = PytorchModel.from_diagrams([dummy_circuit])
    model = model.to(device)

    print("Enter two sentences to compute their similarity.")
    sent1 = input("Sentence 1: ").strip()
    sent2 = input("Sentence 2: ").strip()

    # Compute similarity
    similarity_score = cosine_similarity(sent1, sent2, parser, ansatz, model, device)
    print(f"Similarity score between the sentences: {similarity_score:.4f}")

if __name__ == '__main__':
    main()

