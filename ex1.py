#!/usr/bin/env python
"""
Fine-tuning BERT for paraphrase detection on the MRPC dataset.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC dataset")
    parser.add_argument("--max_train_samples", type=int, default=-1, 
                        help="Max number of training samples to use (-1 for all)")
    parser.add_argument("--max_eval_samples", type=int, default=-1, 
                        help="Max number of evaluation samples to use (-1 for all)")
    parser.add_argument("--max_predict_samples", type=int, default=-1, 
                        help="Max number of prediction samples to use (-1 for all)")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--do_train", action="store_true", 
                        help="Whether to run training")
    parser.add_argument("--do_predict", action="store_true", 
                        help="Whether to run predictions on the test set")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to model to use for predictions")

    return parser.parse_args()

def load_mrpc_dataset(max_train_samples=-1, max_eval_samples=-1, max_predict_samples=-1):
    """
    Load the MRPC dataset from Hugging Face - specifically from nyu-mll/glue
    """
    logger.info("Loading MRPC dataset from nyu-mll/glue...")
    dataset = load_dataset("nyu-mll/glue", "mrpc")

    # Apply sample limits if specified
    if max_train_samples > 0 and max_train_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    if max_eval_samples > 0 and max_eval_samples < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))

    if max_predict_samples > 0 and max_predict_samples < len(dataset["test"]):
        dataset["test"] = dataset["test"].select(range(max_predict_samples))

    logger.info(f"Train: {len(dataset['train'])} examples")
    logger.info(f"Validation: {len(dataset['validation'])} examples")
    logger.info(f"Test: {len(dataset['test'])} examples")

    return dataset

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset for BERT with dynamic padding.
    """
    def tokenize_function(examples):
        # Tokenize the texts with truncation but no padding (dynamic padding)
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding=False,  # Dynamic padding will be applied by the data collator
            max_length=128
        )

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"]
    )

    # Rename label column to labels (the default expected by the models)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    return tokenized_datasets

def train_model(model, tokenizer, train_dataset, eval_dataset, test_dataset, args):
    """
    Train the model and evaluate on validation and test sets.
    """
    # Initialize wandb for tracking
    run_name = f"epochnum{args.num_train_epochs}lr{args.lr}_batchsize{args.batch_size}"
    try:
        import wandb
        wandb.init(project="mrpc-paraphrase-detection", name=run_name)
        use_wandb = True
        logger.info("Successfully initialized Weights & Biases tracking")
    except Exception as e:
        logger.warning(f"Error initializing Weights & Biases: {e}")
        logger.info("Continuing without wandb tracking")
        use_wandb = False

    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.batch_size,
        collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.num_train_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    best_model_state = None
    step_counter = 0
    model_save_path = f"./modelepoch{args.num_train_epochs}lr{args.lr}_batchsize{args.batch_size}"

    # For tracking loss for plotting
    losses = []

    logger.info(f"Starting training on {device}...")

    for epoch in range(args.num_train_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            # Track loss
            current_loss = loss.item()
            losses.append(current_loss)
            epoch_loss += current_loss

            # Log training loss to wandb
            if use_wandb:
                wandb.log({"train/loss": current_loss}, step=step_counter)
            step_counter += 1

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} - Average loss: {avg_epoch_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} - Validation Accuracy: {accuracy:.4f}")

        # Log validation accuracy to wandb
        if use_wandb:
            wandb.log({"validation/accuracy": accuracy}, step=step_counter)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model with accuracy: {best_accuracy:.4f}")
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_labels = []
    
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        test_preds.extend(predictions.cpu().numpy())
        test_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(test_labels, test_preds)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Log test accuracy to wandb
    if use_wandb:
        wandb.log({"test/accuracy": test_accuracy}, step=step_counter)

    # End wandb run
    if use_wandb:
        wandb.finish()

    # Save the best model
    os.makedirs(model_save_path, exist_ok=True)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Saved best model to {model_save_path}")

    # Save results to res.txt
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {best_accuracy:.4f}, test_acc: {test_accuracy:.4f}")
    logger.info(f"Added results to res.txt")

    return best_accuracy, test_accuracy, model_save_path, losses

def generate_predictions(model, tokenizer, test_dataset, args):
    """
    Generate predictions on the test set without padding.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Get original test examples for sentence reconstruction
    original_test = load_dataset("nyu-mll/glue", "mrpc", split="test")
    if args.max_predict_samples > 0:
        original_test = original_test.select(range(min(args.max_predict_samples, len(original_test))))

    # Create data loader - During prediction, we don't pad the samples at all
    # Process one sample at a time to avoid padding
    logger.info("Creating test dataset loader for prediction without padding...")

    all_predictions = []

    # Process each example individually without padding
    for i in range(len(test_dataset)):
        # Get the tokenized input
        item = test_dataset[i]

        # Create tensors
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0)

        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        all_predictions.append(prediction)

    logger.info(f"Generated {len(all_predictions)} predictions")

    # Write predictions to file
    with open("predictions.txt", "w", encoding="utf-8") as f:
        for i, pred in enumerate(all_predictions):
            sentence1 = original_test[i]["sentence1"]
            sentence2 = original_test[i]["sentence2"]
            f.write(f"{sentence1}###{sentence2}###{pred}")

    logger.info("Predictions saved to predictions.txt")
    return all_predictions

def generate_train_loss_plot(loss_data, config):
    """
    Generate a plot for the training loss.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_data)
    plt.title('Training Loss vs Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('train_loss.png')
    logger.info("Generated train_loss.png")

def create_requirements_file():
    """
    Create requirements.txt file.
    """
    with open("requirements.txt", "w") as f:
        f.write("transformers>=4.28.0")
        f.write("datasets>=2.12.0")
        f.write("torch>=2.0.0")
        f.write("wandb>=0.15.0")
        f.write("scikit-learn>=1.2.0")
        f.write("numpy>=1.24.0")
        f.write("matplotlib>=3.5.0")
    logger.info("Created requirements.txt")

def main():
    # Parse arguments
    args = parse_args()

    # Set random seeds for reproducibility
    set_seed(42)

    # Create requirements.txt
    create_requirements_file()

    # Load dataset
    dataset = load_mrpc_dataset(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize dataset
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)

    # Store loss data for plotting
    losses = []

    if args.do_train:
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )

        # Train and evaluate
        best_accuracy, test_accuracy, model_path, training_losses = train_model(
            model,
            tokenizer,
            tokenized_datasets["train"],
            tokenized_datasets["validation"],
            tokenized_datasets["test"],
            args
        )

        # Store losses for plotting
        losses = training_losses

        logger.info(f"Training completed with best validation accuracy: {best_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Generate train loss plot
        generate_train_loss_plot(
            losses, 
            {"epochs": args.num_train_epochs, "lr": args.lr, "batch_size": args.batch_size}
        )

    if args.do_predict:
        # Get model path
        model_path = args.model_path
        if model_path is None:
            # Try to determine best model from res.txt
            try:
                best_model = None
                best_accuracy = 0.0

                with open("res.txt", "r") as f:
                    for line in f:
                        parts = line.strip().split(", ")
                        epoch_num = int(parts[0].split(": ")[1])
                        lr = float(parts[1].split(": ")[1])
                        batch_size = int(parts[2].split(": ")[1])
                        accuracy = float(parts[3].split(": ")[1])

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            model_path = f"./modelepoch{epoch_num}lr{lr}_batchsize{batch_size}"

                logger.info(f"Best model determined from res.txt: {model_path}")
            except Exception as e:
                logger.error(f"Error determining best model: {e}")
                if not args.model_path:
                    logger.error("No model path specified and couldn't determine best model. Exiting.")
                    return

        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} does not exist")
            return

        # Load model for prediction
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Generate predictions
        generate_predictions(model, tokenizer, tokenized_datasets["test"], args)

if __name__ == "__main__":
    main()
