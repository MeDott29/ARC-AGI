import os
import json
import wandb
import torch
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt
from matplotlib import colors
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Any
from pathlib import Path

# Setup color map for visualization
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def load_datasets(data_dir: str = 'arc'):
    """Load all ARC dataset files"""
    files = {
        'training_challenges': 'arc-agi_training_challenges.json',
        'training_solutions': 'arc-agi_training_solutions.json',
        'evaluation_challenges': 'arc-agi_evaluation_challenges.json',
        'evaluation_solutions': 'arc-agi_evaluation_solutions.json',
        'test_challenges': 'arc-agi_test_challenges.json'
    }
    
    data = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        with open(path, 'r') as f:
            data[key] = json.load(f)
    
    return data

def prepare_arc_dataset(challenges: Dict, solutions: Dict = None, is_training: bool = True) -> Dataset:
    """Convert ARC data to HF Dataset format"""
    dataset_items = []
    
    for task_id, task in challenges.items():
        if is_training:
            # Add training pairs
            for train_pair in task['train']:
                dataset_items.append({
                    'input_grid': train_pair['input'],
                    'output_grid': train_pair['output'],
                    'task_id': task_id
                })
            
            # Add test pairs with solutions
            if solutions:
                for i, test_pair in enumerate(task['test']):
                    dataset_items.append({
                        'input_grid': test_pair['input'],
                        'output_grid': solutions[task_id][i],
                        'task_id': task_id
                    })
        else:
            # For test set, only include inputs
            for i, test_pair in enumerate(task['test']):
                dataset_items.append({
                    'input_grid': test_pair['input'],
                    'task_id': task_id,
                    'output_grid': solutions[task_id][i] if solutions else None
                })
    
    return Dataset.from_list(dataset_items)

def compute_grid_accuracy(pred_grid: np.ndarray, true_grid: np.ndarray) -> float:
    """Compute accuracy between predicted and true grids"""
    return np.mean(pred_grid == true_grid)

def collate_arc_data(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Prepare batch data for model"""
    input_grids = [torch.tensor([0] + [x for row in grid for x in row] + [1]) 
                  for grid in examples["input_grid"]]
    
    # Handle output grids if present
    if examples["output_grid"][0] is not None:
        output_grids = [torch.tensor([0] + [x for row in grid for x in row] + [1]) 
                       for grid in examples["output_grid"]]
        
        # Pad sequences
        max_len = max(max(len(x) for x in input_grids), 
                     max(len(x) for x in output_grids))
        
        input_grids = [torch.nn.functional.pad(x, (0, max_len - len(x)), value=1) 
                      for x in input_grids]
        output_grids = [torch.nn.functional.pad(x, (0, max_len - len(x)), value=-100) 
                       for x in output_grids]
        
        return {
            "input_ids": torch.stack(input_grids),
            "labels": torch.stack(output_grids),
            "attention_mask": torch.ones_like(torch.stack(input_grids))
        }
    
    # For inference only
    max_len = max(len(x) for x in input_grids)
    input_grids = [torch.nn.functional.pad(x, (0, max_len - len(x)), value=1) 
                   for x in input_grids]
    
    return {
        "input_ids": torch.stack(input_grids),
        "attention_mask": torch.ones_like(torch.stack(input_grids))
    }

class ArcTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Log training metrics
        if self.state.global_step % self.args.logging_steps == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": self.get_lr(),
                "train/epoch": self.state.epoch,
            })
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_results = super().evaluate(eval_dataset, ignore_keys)
        
        # Generate predictions and compute accuracy
        eval_accuracy = self.compute_accuracy(eval_dataset or self.eval_dataset)
        eval_results["eval_accuracy"] = eval_accuracy
        
        # Log to W&B
        wandb.log({
            "eval/loss": eval_results["eval_loss"],
            "eval/accuracy": eval_accuracy,
            "eval/perplexity": np.exp(eval_results["eval_loss"])
        })
        
        # Visualize some predictions
        self.visualize_predictions(eval_dataset or self.eval_dataset)
        
        return eval_results
    
    def compute_accuracy(self, dataset):
        self.model.eval()
        accuracies = []
        
        for i in range(0, len(dataset), self.args.eval_batch_size):
            batch = dataset[i:i + self.args.eval_batch_size]
            batch = collate_arc_data(batch)
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=512,
                    pad_token_id=1
                )
            
            for j, pred_tokens in enumerate(outputs):
                pred_grid = self.tokens_to_grid(pred_tokens.cpu().numpy())
                true_grid = batch["labels"][j].cpu().numpy()
                true_grid = self.tokens_to_grid(true_grid[true_grid != -100])
                accuracies.append(compute_grid_accuracy(pred_grid, true_grid))
        
        return np.mean(accuracies)
    
    def visualize_predictions(self, dataset, num_samples=5):
        self.model.eval()
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        for idx in indices:
            batch = collate_arc_data([dataset[idx]])
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=512,
                    pad_token_id=1
                )
            
            pred_grid = self.tokens_to_grid(outputs[0].cpu().numpy())
            true_grid = self.tokens_to_grid(batch["labels"][0][batch["labels"][0] != -100].cpu().numpy())
            input_grid = self.tokens_to_grid(batch["input_ids"][0][batch["input_ids"][0] != 1].cpu().numpy())
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            ax1.imshow(input_grid, cmap=cmap, norm=norm)
            ax1.set_title("Input Grid")
            ax1.grid(True)
            
            ax2.imshow(pred_grid, cmap=cmap, norm=norm)
            ax2.set_title(f"Predicted Grid\nAccuracy: {compute_grid_accuracy(pred_grid, true_grid):.2%}")
            ax2.grid(True)
            
            ax3.imshow(true_grid, cmap=cmap, norm=norm)
            ax3.set_title("True Grid")
            ax3.grid(True)
            
            plt.tight_layout()
            wandb.log({f"predictions/sample_{idx}": wandb.Image(plt)})
            plt.close()
    
    @staticmethod
    def tokens_to_grid(tokens):
        """Convert token sequence back to grid"""
        # Remove special tokens
        grid_values = tokens[1:-1]  # Remove start/end tokens
        grid_size = int(np.sqrt(len(grid_values)))
        return np.array(grid_values).reshape(grid_size, grid_size)

def main():
    # Initialize wandb
    wandb.init(project="arcopt")
    
    # Load datasets
    data = load_datasets()
    
    # Prepare datasets
    train_dataset = prepare_arc_dataset(
        data['training_challenges'], 
        data['training_solutions']
    )
    eval_dataset = prepare_arc_dataset(
        data['evaluation_challenges'], 
        data['evaluation_solutions']
    )
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/galactica-125m",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./arc-model-checkpoints",
        num_train_epochs=5000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        neftune_noise_alpha=0.1,
        warmup_steps=100,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        report_to="wandb",
        optim="adamw_bnb_8bit",
    )
    
    # Initialize trainer
    trainer = ArcTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_arc_data,
    )
    
    # Train model
    trainer.train()
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    print("Final evaluation metrics:", final_metrics)
    
    # Save final model
    trainer.save_model("./arc-model-final")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()