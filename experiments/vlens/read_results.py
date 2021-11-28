import os
from experiments.data.load_datasets import load_vector_mnist
from experiments.vlens.mnist_sum import ResNetMNISTsum
from torch_explain.nn import semantics


def main():
    train_data, test_data, concept_names, label_names = load_vector_mnist('../data')

    result_dir = './results/mnist_sum/'
    model = ResNetMNISTsum.load_from_checkpoint(os.path.join(result_dir, 'model.pt'),
                                                map_location="cuda")
    model.freeze()

    c1 = train_data.tensors[2]
    c2 = train_data.tensors[3]
    y = train_data.tensors[4]
    c1_emb, c2_emb, c_emb, y_emb = model(train_data.tensors[0], train_data.tensors[1])
    c1_logits = semantics(c1_emb)
    c2_logits = semantics(c2_emb)
    y_logits = semantics(y_emb)

    # compute accuracy
    c_accuracy = (c1_logits > 0.5).eq(c1).sum().item() + (c2_logits > 0.5).eq(c2).sum().item()
    c_accuracy = c_accuracy / (c1.size(0) * c1.size(1) * 2)
    y_accuracy = (y_logits > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
    print(f'c-accuracy: {c_accuracy:.4f}, y-accuracy: {y_accuracy:.4f}')


if __name__ == '__main__':
    main()
