import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss
from torch_explain.nn.concepts import ConceptReasoningLayer

class ImageEmbedder(nn.Module):
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

class ConceptClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 5),
            nn.ReLU(),
            nn.Linear(5, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class TupleCreator(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x, t):
        # x = torch.gather(x, dim=0, index=t)
        x = x[t]
        x = x.view([t.shape[0], -1])
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


class ManifoldRelationalDCR(pl.LightningModule):
    def __init__(self, input_features, emb_size, manifold_arity, num_classes, predict_relation = False, concept_names=None, explanations=None, learning_rate=0.001,  temperature=10, verbose: bool = False, gpu=True):
        super().__init__()
        self.image_embedder = ImageEmbedder(input_features, emb_size)
        self.concept_classifier = ConceptClassifier(emb_size, num_classes=num_classes)
        self.tuple_creator = TupleCreator()
        self.emb_size = emb_size
        self.predict_relation = predict_relation
        if self.predict_relation:
            self.relation_classifier = ConceptClassifier(emb_size*2, num_classes=1) #only-binary TODO: n-ary, multiple relations
            self.reasoner = ConceptReasoningLayer(emb_size*manifold_arity, n_concepts=num_classes+1, n_classes = num_classes) # +1 for the relation
        else:
            self.reasoner = ConceptReasoningLayer(emb_size*manifold_arity, n_concepts=num_classes, n_classes = num_classes)


        self.concept_names = concept_names
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")



    def forward(self, X, body, head):
        embeddings = self.image_embedder(X)
        t = torch.concat((body, head), dim=-1)
        emb_tuple = self.tuple_creator(embeddings, t)


        concepts = self.concept_classifier(embeddings)
        concepts_body = self.tuple_creator(concepts, body)

        if self.predict_relation:
            relations = self.relation_classifier(emb_tuple)
            concepts_body = torch.concat((concepts_body, relations), dim=-1)


        tasks = self.reasoner(emb_tuple, concepts_body)

        if self.predict_relation:
            return concepts, relations, tasks
        else:
            return concepts, tasks

    def training_step(self, I, batch_idx):
        pass

    def validation_step(self, I, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


