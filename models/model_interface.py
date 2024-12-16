import pytorch_lightning as pl
import torch
import importlib
import inspect
from MyLoss.loss_factory import create_loss
from MyOptimizer.optim_factory import create_optimizer
import torchmetrics
import pandas as pd
class ModelInterface(pl.LightningModule):

    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']


        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.valdata = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.testdata = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        #---->epoch end
        self.train_epoch_outputs = []
        self.validation_epoch_outputs = []
        self.test_epoch_outputs = []
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(task='multiclass', num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='multiclass', num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(task='multiclass', num_classes = self.n_classes),
                                                     torchmetrics.F1Score(task='multiclass', num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task='multiclass', average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(task='multiclass', average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(task='multiclass', average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(task='binary', num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = 2, average = 'micro'),
                                                     torchmetrics.CohenKappa(task='binary', num_classes = 2),
                                                     torchmetrics.F1Score(task='binary', num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task='binary', average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(task='binary', average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')
    def training_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        loss = self.loss(logits, label)
        self.train_epoch_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})

        return {'loss': loss} 

    def on_train_epoch_end(self):
        labels = torch.cat([x['label'] for x in self.train_epoch_outputs], dim = 0)
        preds = torch.cat([x['Y_hat'] for x in self.train_epoch_outputs], dim = 0)
        for i in range(len(labels)):
            self.data[int(labels[i])]['count'] += 1
            if int(labels[i]) == int(preds[i]):
                self.data[int(labels[i])]['correct'] += 1
        #binary_acc(self.data, labels, preds)
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print(f'class {c}: acc {acc}, correct {correct}/{count}')
        #---->clear
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.train_epoch_outputs.clear()
    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        self.validation_epoch_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})


    def on_validation_epoch_end(self):
        torch.use_deterministic_algorithms(False)
        logits = torch.cat([x['logits'] for x in self.validation_epoch_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in self.validation_epoch_outputs], dim = 0)
        max_probs = torch.cat([x['Y_hat'] for x in self.validation_epoch_outputs])
        target = torch.cat([x['label'] for x in self.validation_epoch_outputs], dim = 0)
        
        #---->
        self.log('val_loss', self.loss(logits, target), prog_bar=True, on_epoch=True, logger=True)
        if self.n_classes == 2:
            self.log('auc', self.AUROC(probs[:,1], target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        else:
            self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()), on_epoch = True, logger = True)
        #---->log acc
        for i in range(len(target)):
            self.valdata[int(target[i])]['count'] += 1
            if int(target[i]) == int(max_probs[i]):
                self.valdata[int(target[i])]['correct'] += 1
        for c in range(self.n_classes):
            count = self.valdata[c]["count"]
            correct = self.valdata[c]["correct"]
            if count == 0: 
                acc = 0.0
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
            self.log('acc{}'.format(c), acc, prog_bar=True, on_epoch=True, logger=True)
        #---->clear
        self.valdata = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.validation_epoch_outputs.clear()

    def test_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        self.test_epoch_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})


    def on_test_epoch_end(self):
        torch.use_deterministic_algorithms(False)
        probs = torch.cat([x['Y_prob'] for x in self.test_epoch_outputs], dim = 0)
        max_probs = torch.cat([x['Y_hat'] for x in self.test_epoch_outputs])
        target = torch.cat([x['label'] for x in self.test_epoch_outputs], dim = 0)
        
        #---->
        if self.n_classes == 2:
            auc = self.AUROC(probs[:,1], target.squeeze())
        else:
            auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()

        #---->log acc
        for c in range(self.n_classes):
            count = self.testdata[c]["count"]
            correct = self.testdata[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
            metrics['acc{}'.format(c)] = acc
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')
        #----> clear
        self.test_epoch_outputs.clear()
        self.testdata = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        #scheduler = StepLR(optimizer, step_size=self.hparams.scheduler.step_size, gamma=self.hparams.scheduler.gamma)
        return [optimizer]

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)


