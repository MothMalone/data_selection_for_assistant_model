from transformers import Trainer
import torch

class TrainerWithTrainingDynamics(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_dynamics = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return a tuple (loss, ...)
        where the first element is the loss.
        """
        # Save labels
        labels = inputs.pop("labels")
        
        # Save and remove indices - the model doesn't expect this
        indices = None
        if "index" in inputs:
            indices = inputs.pop("index")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        # Log dynamics
        if self.is_in_train and indices is not None:
            self.log_training_dynamics(indices, logits)

        return (loss.mean(), outputs) if return_outputs else loss.mean()

    def log_training_dynamics(self, indices, logits):
        # Initialize epoch dynamics if needed
        if not hasattr(self, 'epoch_dynamics'):
            self.epoch_dynamics = {}
        
        # Log logits for each example
        for i, index in enumerate(indices):
            self.epoch_dynamics[int(index)] = logits[i].detach().cpu().numpy()

    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        if hasattr(self, 'epoch_dynamics'):
            self.training_dynamics.append(self.epoch_dynamics)
            self.epoch_dynamics = {}

    def get_training_dynamics(self):
        return self.training_dynamics
