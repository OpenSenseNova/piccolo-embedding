import os
import torch
from transformers.trainer import Trainer
from transformers.trainer import logger, Optional

class STETrainer(Trainer):
    def __init__(self, use_optimum, efficient_save, **kwargs):
        super().__init__(**kwargs)
        self.use_optimum = use_optimum
        self.efficient_save = efficient_save

    def _save(self, output_dir: Optional[str] = None, **kwargs):
        '''save the unwrap model, bcz we use better transformer'''
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", output_dir)
        if self.use_optimum:
            from optimum.bettertransformer import BetterTransformer
            unwrap_model = BetterTransformer.reverse(self.model.embedder.encoder)
        else:
            unwrap_model = self.model.embedder.encoder
        if self.is_world_process_zero():
            unwrap_model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
            self.tokenizer.save_pretrained(output_dir)
            if hasattr(self.model, 'scaling_layer'):
                scaling_layer_sd_st = {'linear.weight': self.model.scaling_layer.state_dict()['linear.weight'].data.cpu(), 
                                    'linear.bias': self.model.scaling_layer.state_dict()['linear.bias'].data.cpu()}
                torch.save(scaling_layer_sd_st, os.path.join(output_dir, 'scaling_layer_st.bin'))


    def _save_checkpoint(self, model, trial, metrics=None):
        if self.efficient_save:
            '''only save the model ckpt weights to save disk mem'''
            from transformers.trainer import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
        else:
            super()._save_checkpoint(model, trial, metrics)
