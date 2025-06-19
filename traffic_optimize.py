import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from traffic import EPOCHS, IMG_HEIGHT, IMG_WIDTH, NUM_CATEGORIES, TEST_SIZE, build_conv_pooling_layers, build_hidden_layers, load_data, get_model
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import sys

def objective(trial):
    """
    Objective function for Bayesian optimization.
    Optuna will call this function many times with different parameter suggestions.
    """
    
    # Let Optuna suggest hyperparameters
    params = {
        # Convolutional layer parameters
        'num_layers_conv': trial.suggest_int('num_layers_conv', 1, 3),
        'nodes_per_conv_layer': trial.suggest_categorical('nodes_per_conv_layer', [16, 32, 64, 128, 256, 512]),
        'kernel_size_x': trial.suggest_categorical('kernel_size_x', [3, 5]),
        'kernel_size_y': trial.suggest_categorical('kernel_size_y', [3, 5]),
        'pool_size': trial.suggest_categorical('pool_size', [2, 3]),
        
        # Hidden layer parameters  
        'num_layers_hidden': trial.suggest_int('num_layers_hidden', 1, 5),
        'first_hidden_layer_nodes': trial.suggest_categorical('first_hidden_layer_nodes', [64, 128, 256, 512]),
        'subsequent_hidden_layer_nodes_decrease': trial.suggest_float('subsequent_hidden_layer_nodes_decrease', 0.25, 0.75),
        'first_hidden_layer_dropout': trial.suggest_float('first_hidden_layer_dropout', 0.25, 0.75),
        'subsequent_hidden_layer_dropout_decrease': trial.suggest_float('subsequent_hidden_layer_dropout_decrease', 0.05, 0.50),
        
        # Training parameters
        'epochs': trial.suggest_int('epochs', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    }
    
    try:
        # Create validation split (don't use test set for hyperparameter search!)
        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )
        
        # Build model with suggested parameters
        model = build_optimized_model(params)
        
        # Custom optimizer with suggested learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']), # type: ignore
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Train model with early stopping and validation monitoring
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=3, 
                restore_best_weights=True
            ),
            # Optuna pruning callback (stops bad trials early)
            optuna.integration.KerasPruningCallback(trial, 'val_accuracy')
        ]
        
        history = model.fit(
            x_train_split, y_train_split,
            epochs=params['epochs'],
            validation_data=(x_val_split, y_val_split),
            callbacks=callbacks
        )
        
        # Return validation accuracy (what we want to maximize)
        val_accuracy = max(history.history['val_accuracy'])
        
        # Log this trial's results
        print(f"Trial {trial.number}: Val Accuracy = {val_accuracy:.4f}")
        print(f"  Conv layers: {params['num_layers_conv']}, nodes: {params['nodes_per_conv_layer']}")
        print(f"  Hidden layers: {params['num_layers_hidden']}, nodes: {params['first_hidden_layer_nodes']}")
        print(f"  Kernel: ({params['kernel_size_x']},{params['kernel_size_y']}), Epochs: {params['epochs']}")
        
        return val_accuracy
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return very low score for failed trials

def build_optimized_model(params):
    """
    Build model using parameters suggested by Optuna.
    """
    
    # Build conv layers
    conv_layers = build_conv_pooling_layers(
        num_layers_conv=params['num_layers_conv'],
        nodes_per_conv_layer=params['nodes_per_conv_layer'],
        kernel_size=(params['kernel_size_x'], params['kernel_size_y']),
        activation="relu",
        pool_size=(params['pool_size'], params['pool_size'])
    )
    
    # Build hidden layers
    hidden_layers = build_hidden_layers(
        num_layers_hidden=params['num_layers_hidden'],
        first_hidden_layer_nodes=params['first_hidden_layer_nodes'],
        subsequent_hidden_layer_nodes_decrease=params['subsequent_hidden_layer_nodes_decrease'],
        hidden_layer_activation_algo="relu",
        first_hidden_layer_dropout=params['first_hidden_layer_dropout'],
        subsequent_hidden_layer_dropout_decrease=params['subsequent_hidden_layer_dropout_decrease']
    )
    
    # Create complete model
    model = keras.models.Sequential([
        keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        *conv_layers,
        keras.layers.Flatten(),
        *hidden_layers,
        keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    return model

def run_hyperparameter_optimization(n_trials=10):
    """
    Run Bayesian hyperparameter optimization.
    """
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',  # We want to maximize accuracy
        sampler=TPESampler(seed=42),  # Bayesian optimization algorithm
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)  # Stop bad trials early
    )
    
    print(f"Starting Bayesian optimization with {n_trials} trials...")
    print("This will take several hours. Go get coffee! ☕")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=None)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    
    print(f"\nBest validation accuracy: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Test best model on actual test set
    print("\n" + "-"*40)
    print("Testing best model on test set...")
    
    best_model = build_optimized_model(study.best_params)
    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=study.best_params['learning_rate']), # type: ignore
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    best_model.fit(x_train, y_train, epochs=study.best_params['epochs'])
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    print(f"Improvement over baseline: {test_accuracy - baseline_accuracy:.4f}")
    
    return study, best_model

# Usage example:
def main_with_optimization():
    """
    Main function that includes hyperparameter optimization.
    """
    global x_train, y_train, x_test, y_test, baseline_accuracy
    
    # Load and prepare data (your existing code)
    images, labels = load_data(sys.argv[1])
    labels = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    
    # Get baseline accuracy with your current model
    print("Getting baseline accuracy...")
    baseline_model = get_model()  # Your current model
    baseline_model.fit(x_train, y_train, epochs=EPOCHS)
    _, baseline_accuracy = baseline_model.evaluate(x_test, y_test)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Run optimization
    study, best_model = run_hyperparameter_optimization(n_trials=10)
    
    # Save best model
    if len(sys.argv) == 3:
        filename = sys.argv[2].replace('.h5', '_optimized.keras')
        best_model.save(filename)
        print(f"Optimized model saved to {filename}")
    
    return study, best_model

if __name__ == "__main__":
    main_with_optimization()