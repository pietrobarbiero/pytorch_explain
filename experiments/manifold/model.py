import torch
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss, ModuleList

from torch_explain.logic.indexing import group_by_no_for
from torch_explain.nn.concepts import ConceptReasoningLayer


class ManifoldRelationalDCR(pl.LightningModule):
    def __init__(self, indexer, input_features, emb_size, manifold_arity, num_classes, num_relations,
                 predict_relation = False, set_level_rules=False, crisp=False,
                 concept_names=None, explanations=None, learning_rate=0.01,  temperature=10,
                 verbose: bool = False, gpu=True):
        super().__init__()
        self.indexer = indexer
        self.emb_size = emb_size
        self.predict_relation = predict_relation
        self.set_level_rules = set_level_rules
        self.concept_names = concept_names
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")

        # neural nets
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
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
            self.relation_embedders = ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                ),  # q(X) classifier
                torch.nn.Sequential(
                    torch.nn.Linear(emb_size * 2, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                ),  # r(X,Y) classifier
            ])
        else:
            self.relation_classifiers = ModuleList([
                torch.nn.Sequential(torch.nn.Linear(emb_size, 1), torch.nn.Sigmoid()),  # q(X) classifier
            ])
            self.relation_embedders = ModuleList([
                torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size)),  # q(X) classifier
            ])
        self.reasoner = ConceptReasoningLayer(emb_size * manifold_arity, n_concepts=num_relations, n_classes=num_classes, set_level_rules=set_level_rules)

    def forward(self, X, explain=False):
        X = X.squeeze(0)

        embeddings = self.encoder(X)

        # relation/concept predictions
        preds_rel, embs_rel, queries_ids = [], [], []
        for rel_id, (relation_classifier, relation_embedder) in enumerate(zip(self.relation_classifiers, self.relation_embedders)):
            embeding_constants, constants_index, query_index = self.indexer.apply_index(embeddings, 'atoms', rel_id)
            preds_rel.append(relation_classifier(embeding_constants))
            embs_rel.append(relation_embedder(embeding_constants))
            queries_ids.append(query_index)
        queries_ids = torch.cat(queries_ids, dim=0)
        preds_rel = torch.cat(preds_rel, dim=0)
        embs_rel = torch.cat(embs_rel, dim=0)

        # task predictions
        preds_xformula, index_xformula, formula_ids = self.indexer.apply_index(preds_rel, 'formulas', 0)
        embed_xformula, index_xformula, formula_ids = self.indexer.apply_index(embs_rel, 'formulas', 0)
        y_preds = self.reasoner(embed_xformula, preds_xformula)

        # TODO: create OR rule for the task predictions
        # aggregate task predictions (next: do it with OR)
        y_preds_group = group_by_no_for(groupby_values=formula_ids, tensor_to_group=y_preds)
        y_preds_group = torch.stack(y_preds_group, dim=0)
        y_preds_mean = y_preds_group.mean(dim=1)

        explanations = None
        if explain:
            explanations = self.reasoner.explain(embed_xformula, preds_xformula, 'global')

        return preds_rel, y_preds, y_preds_mean, explanations, queries_ids, formula_ids

    def training_step(self, I, batch_idx):
        X, q_labels = I
        q_labels = q_labels.squeeze(0)

        c_pred, y_preds, y_pred_agg, explanations, queries_ids, formula_ids = self.forward(X)

        # get supervised slice
        c_pred_sup = self.indexer.get_supervised_slice(c_pred, queries_ids)
        q_labels_sup = self.indexer.get_supervised_slice(q_labels, torch.unique(formula_ids))
        q_labels_sup_all = q_labels_sup.squeeze().repeat(len(torch.unique(formula_ids)))

        concept_loss = self.bce(c_pred_sup, q_labels.float())
        task_loss = self.cross_entropy(y_pred_agg, q_labels_sup.squeeze())
        task_loss += self.cross_entropy(y_preds, q_labels_sup_all.squeeze())
        loss = concept_loss + 0.5*task_loss

        task_accuracy = accuracy_score(q_labels_sup.squeeze(), y_pred_agg[:, 1] > 0.5)
        task_accuracy_all = accuracy_score(q_labels_sup_all, y_preds[:, 1] > 0.5)
        concept_accuracy = accuracy_score(q_labels, c_pred_sup > 0.5)
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f} task accuracy (all): {task_accuracy_all:.4f} ')
        return loss

    def validation_step(self, I, batch_idx):
        X, q_labels = I
        q_labels = q_labels.squeeze(0)

        c_pred, y_preds, y_pred_agg, explanations, queries_ids, formula_ids = self.forward(X)

        # get supervised slice
        c_pred_sup = self.indexer.get_supervised_slice(c_pred, queries_ids)
        q_labels_sup = self.indexer.get_supervised_slice(q_labels, torch.unique(formula_ids))

        concept_loss = self.bce(c_pred_sup, q_labels.float())
        task_loss = self.cross_entropy(y_pred_agg, q_labels_sup.squeeze())
        loss = concept_loss + 0.5*task_loss
        return loss

    def test_step(self, I, batch_idx):
        X, q_labels = I
        q_labels = q_labels.squeeze(0)

        c_pred, y_preds, y_pred_agg, explanations, queries_ids, formula_ids = self.forward(X)

        # get supervised slice
        c_pred_sup = self.indexer.get_supervised_slice(c_pred, queries_ids)
        q_labels_sup = self.indexer.get_supervised_slice(q_labels, torch.unique(formula_ids))

        task_accuracy = accuracy_score(q_labels_sup.squeeze(), y_pred[:, 1] > 0.5)
        concept_accuracy = accuracy_score(q_labels, c_pred_sup > 0.5)
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')
        print(explanations)
        return task_accuracy, concept_accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

