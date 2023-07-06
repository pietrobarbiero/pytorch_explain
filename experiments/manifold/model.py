import torch
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss, ModuleList

from torch_explain.logic.indexing import group_by_no_for
from torch_explain.nn.concepts import ConceptReasoningLayer


class ManifoldRelationalDCR(pl.LightningModule):
    def __init__(self, indexer, input_features, emb_size, manifold_arity, num_classes, num_relations,
                 predict_relation = False, set_level_rules=False, crisp=False, task_names=None,
                 concept_names=None, explanations=None, learning_rate=0.01,  temperature=10,
                 verbose: bool = False, gpu=True):
        super().__init__()
        self.indexer = indexer
        self.emb_size = emb_size
        self.predict_relation = predict_relation
        self.set_level_rules = set_level_rules
        self.concept_names = concept_names#
        self.task_names = task_names
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")

        # neural nets
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LeakyReLU(),
        )
        if self.predict_relation:
            self.relation_classifiers = ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, 1),
                    torch.nn.Sigmoid()
                ),  # q(X) classifier
                torch.nn.Sequential(
                    torch.nn.Linear(emb_size * 2, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, 1),
                    torch.nn.Sigmoid()
                ),  # r(X,Y) classifier
            ])
        else:
            self.relation_classifiers = ModuleList([
                torch.nn.Sequential(torch.nn.Linear(emb_size, 1), torch.nn.Sigmoid()),  # q(X) classifier
            ])
        self.reasoner = ConceptReasoningLayer(emb_size * manifold_arity, n_concepts=num_relations,
                                              n_classes=num_classes, set_level_rules=set_level_rules, temperature=temperature)

    def forward(self, X, explain=False):
        X = X.squeeze(0)

        embeddings = self.encoder(X)

        # relation/concept predictions
        concept_predictions = []
        for rel_id, relation_classifier in enumerate(self.relation_classifiers):
            embedding_constants, _, _ = self.indexer.apply_index_atoms(embeddings, rel_id)
            concept_predictions.append(relation_classifier(embedding_constants))
        concept_predictions = torch.cat(concept_predictions, dim=0)

        # task predictions
        if torch.any(torch.isnan(concept_predictions)):
            print()
        embed_substitutions, preds_xformula = self.indexer.apply_index_formulas(embeddings, concept_predictions, 'phi')
        grounding_preds = self.reasoner(embed_substitutions, preds_xformula)#, sign_attn=torch.ones(preds_xformula.shape[0], preds_xformula.shape[1], 1))

        # aggregate task predictions (next: do it with OR)
        task_predictions = self.indexer.group_or(grounding_preds, 'phi')
        task_predictions = torch.max(task_predictions, concept_predictions)
        # task_predictions = torch.mean(torch.cat((task_predictions, concept_predictions), dim=1), dim=1).unsqueeze(1)

        c_preds = self.indexer.lookup_query(concept_predictions, 'concepts')
        y_preds = self.indexer.lookup_query(task_predictions, 'tasks')

        explanations = None
        if explain:
            explanations = self.reasoner.explain(embed_substitutions, preds_xformula, 'global',
                                                 self.concept_names, self.task_names,
                                                 sign_attn=torch.ones(preds_xformula.shape[0], preds_xformula.shape[1], 1))

        return c_preds, y_preds, explanations

    def training_step(self, I, batch_idx):
        X, q_labels = I
        q_labels = q_labels.squeeze(0)

        c_preds, y_preds, explanations = self.forward(X)

        c_true = self.indexer.lookup_query(q_labels, 'concepts')
        y_true = self.indexer.lookup_query(q_labels, 'tasks')

        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce(y_preds, y_true.float())
        loss = concept_loss + 0.5*task_loss

        task_accuracy = accuracy_score(y_true.squeeze(), y_preds > 0.5)
        concept_accuracy = accuracy_score(c_true, c_preds > 0.5)
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, q_labels = I
        q_labels = q_labels.squeeze(0)

        c_preds, y_preds, explanations = self.forward(X)

        c_true = self.indexer.lookup_query(q_labels, 'concepts')
        y_true = self.indexer.lookup_query(q_labels, 'tasks')
        if torch.any(torch.logical_or(c_preds<0, c_preds>1)) or torch.any(torch.logical_or(y_preds<0, y_preds>1)):
            print()
        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce(y_preds, y_true.float())
        loss = concept_loss + 0.5*task_loss
        return loss

    def test_step(self, I, batch_idx):
        X, q_labels = I
        q_labels = q_labels.squeeze(0)

        c_preds, y_preds, explanations = self.forward(X, explain=True)

        c_true = self.indexer.lookup_query(q_labels, 'concepts')
        y_true = self.indexer.lookup_query(q_labels, 'tasks')

        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce(y_preds, y_true.float())
        loss = concept_loss + 0.5*task_loss

        task_accuracy = accuracy_score(y_true.squeeze(), y_preds > 0.5)
        concept_accuracy = accuracy_score(c_true, c_preds > 0.5)
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')
        print(explanations)
        print(f'C size: {len(c_true)}, C pos: {c_true.sum()} C neg: {len(c_true)-c_true.sum()}')
        print(f'Y size: {len(y_true)}, Y pos: {y_true.sum()} Y neg: {len(y_true)-y_true.sum()}')
        return task_accuracy, concept_accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

