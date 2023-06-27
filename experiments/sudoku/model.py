import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss
from torch_explain.nn.concepts import ConceptReasoningLayer
from sklearn.metrics import accuracy_score



class MNISTEncoder(nn.Module):
    def __init__(self, emb_size):
        super(MNISTEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.linear = nn.Linear(16 * 4 * 4, emb_size)
        self.emb_size = emb_size


    def forward(self, x):
        sh = len(x.shape)
        if sh>4:
            first_dims = x.shape[:-3]
            x = x.view(-1, *x.shape[2:])
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.linear(x)
        if sh>4:
            x = x.view(*first_dims, self.emb_size)
        return x




class ConceptClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 5),
            nn.ReLU(),
            nn.Linear(5, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        sh = len(x.shape)
        if sh > 2:
            first_dims = x.shape[:-1]
            x = x.view(-1, x.shape[-1])
        x = self.encoder(x)
        if sh > 2:
            x = x.view(*first_dims, self.num_classes)
        return x


class TupleEmbedder(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 5),
            nn.ReLU(),
            nn.Linear(5, output_features),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x


class SudokuRelationalDCR(pl.LightningModule):
    def __init__(self, emb_size, manifold_arity, manifolds, num_classes, concept_names=None, explanations=None, learning_rate=0.001,  temperature=10, verbose: bool = False, gpu=True):
        super().__init__()
        self.image_embedder = MNISTEncoder(emb_size)
        self.concept_classifier = ConceptClassifier(emb_size, num_classes=num_classes)
        self.emb_size = emb_size
        self.reasoner = ConceptReasoningLayer(emb_size*manifold_arity, n_concepts=num_classes*manifold_arity, n_classes = 1)
        self.manifolds = manifolds

        self.concept_names = concept_names
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = CrossEntropyLoss(reduction="mean")



    def forward(self, X, explain = False):

        # e.g. n = 4
        # X is [num_boards, num_cells, 1, 28, 28], e.g. [100, 16, 1, 28, 28]
        # manifolds [num_columns + num_rows + num_squares, n]; e.g. [4+4+4,4]
        # let's call num_columns + num_rows + num_squares = num_manifolds


        # How rows and columsn and squares are positioned does not depend on the example. So, we have them stored in the object
        manifolds = self.manifolds

        # We embed each image in each board [num_boards, num_cells, emb_size]
        embeddings = self.image_embedder(X)
        #We compute the embeddings of the tuple of images in each manifold [num_boards, num_manifolds, n*emb_size]
        emb_tuple = embeddings[:,manifolds].view(X.shape[0], manifolds.shape[0], -1) # concat embeddings of each manifold (square, column, rule)


        # We compute the concepts for each image in each board [num_boards, num_cells, n]
        concepts = self.concept_classifier(embeddings)
        # We compute the concepts of the tuple of images in each manifold [num_boards, num_manifolds, n*n]
        concepts_tuple = concepts[:,manifolds].view(concepts.shape[0], manifolds.shape[0], -1) # concat embeddings of each manifold (square, column, rule)


        # Since self.reasoner works at the level of groundings, we put the groundings of all the boards together (TODO: this is anti-DeepSet; in DeepSet you want to use this dimension)
        emb_tuple_flat = emb_tuple.view(-1, emb_tuple.shape[-1])
        concepts_tuple_flat = concepts_tuple.view(-1, concepts_tuple.shape[-1])

        # We compute the single concept (aka "all_different(X)") for all the groundigns of all the boards
        # Remember a grounding here is either a row, a columsn or a square
        tasks = self.reasoner(emb_tuple_flat, concepts_tuple_flat)

        # We make explicit again which are the groundings of each board
        tasks = tasks.view(*emb_tuple.shape[0:2])

        # We ask for an and (product here) of all the concepts TODO: we are adding the second level rule as ground truth
        # Here we are writing the following eule:
        # valid_sudoku <- all_different(row1) and all_different(row2) and .... and all_different(square_n)
        tasks = torch.prod(tasks, dim=-1, keepdim=True)

        explanations = None
        if explain:
            explanations = self.reasoner.explain(emb_tuple_flat, concepts_tuple_flat, 'global')

        return concepts, tasks, explanations

    def training_step(self, I, batch_idx):

        X, c_train, y_train = I


        c_pred, y_pred, _ = self.forward(X)

        concept_loss = self.bce(c_pred, c_train)
        task_loss = self.bce(y_pred, y_train)
        loss = concept_loss + 0.5 * task_loss
        return loss

    def validation_step(self, I, batch_idx):

        X, c_train, y_train = I


        c_pred, y_pred, explanations = self.forward(X, explain=False)

        concept_loss = self.bce(c_pred, c_train)
        task_loss = self.bce(y_pred, y_train)
        loss = concept_loss + 0.5 * task_loss

        print(explanations)
        return loss

    def test_step(self, I, batch_idx):

        X, c_train, y_train = I
        c_pred, y_pred, explanations = self.forward(X)

        # c_pred, y_pred, explanations = self.forward(X, explain=True)

        task_accuracy = accuracy_score(y_train, y_pred > 0.5)
        concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
        # print(
        #     f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')
        # print(explanations)
        return task_accuracy, concept_accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


