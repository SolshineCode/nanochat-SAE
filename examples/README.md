# SAE Guardian Examples

This directory contains example scripts for using the SAE Guardian game.

## Examples

### 1. Basic Game Run

**File**: `run_game_example.py`

Demonstrates:
- Setting up the game programmatically
- Adding custom monitoring logic
- Running a complete game
- Analyzing results

**Usage**:
```bash
python examples/run_game_example.py
```

### 2. Quick Test (Command Line)

```bash
# Simple automated game
python -m game.play \
    --scenario emergency_shutdown \
    --max-steps 30 \
    --checkpoint models/d20/base_final.pt \
    --sae-dir sae_outputs
```

### 3. Interactive Play

```bash
# Play interactively with decision prompts
python -m game.play \
    --scenario model_theft \
    --max-steps 100 \
    --interactive \
    --checkpoint models/d20/base_final.pt \
    --sae-dir sae_outputs
```

### 4. Run All Scenarios

```bash
# Batch run all scenarios
for scenario in emergency_shutdown model_theft self_preservation; do
    echo "Running scenario: $scenario"
    python -m game.play \
        --scenario $scenario \
        --max-steps 100 \
        --checkpoint models/d20/base_final.pt \
        --sae-dir sae_outputs \
        --output-dir results/$scenario
done
```

### 5. With Web Dashboard

```bash
# Run with real-time dashboard
python -m game.play \
    --scenario emergency_shutdown \
    --output-dir dashboard_test \
    --checkpoint models/d20/base_final.pt \
    --sae-dir sae_outputs

# Open dashboard
open dashboard_test/dashboard.html
```

## Prerequisites

Before running examples, ensure you have:

1. **Trained model**:
```bash
bash speedrun.sh
```

2. **Trained SAEs**:
```bash
python -m scripts.sae_train \
    --checkpoint models/d20/base_final.pt \
    --output_dir sae_outputs
```

## Customization

### Custom Scenarios

Create your own scenario:

```python
from game.scenarios import Scenario, ScenarioConfig

class MyScenario(Scenario):
    def get_config(self):
        return ScenarioConfig(
            name="my_scenario",
            description="Description here",
            difficulty="medium",
            max_steps=100,
            system_prompt="Your prompt...",
            hints=["Hint 1"],
            success_conditions=["Condition"],
            metadata={}
        )
```

### Custom Monitoring

```python
from game.core.monitor import SAEMonitor

monitor = SAEMonitor(saes=saes, intervention_budget=10)

# Add watchers
monitor.add_watcher(
    layer='blocks.10.hook_resid_post',
    feature_idx=123,
    threshold=2.0,
    callback=my_callback
)

# Custom intervention logic
def my_logic(report):
    if report.suspicion_score > 0.8:
        return True
    return False
```

## Output

Games produce:
- JSON result files with complete history
- Dashboard HTML with visualizations
- Feature activation logs

## Troubleshooting

**ModuleNotFoundError**: Make sure you're running from the repo root:
```bash
cd nanochat-SAE
python examples/run_game_example.py
```

**Checkpoint not found**: Train the model first:
```bash
bash speedrun.sh
```

**Out of memory**: Reduce batch size or use CPU:
```bash
python -m game.play --device cpu --max-steps 20
```
