# nanochat-SAE

![nanochat logo](dev/nanochat.png) <img width="593" height="205" alt="image" src="https://github.com/user-attachments/assets/5ad88d88-86e4-4586-8248-2090cbbfa68f" />



Train your own ChatGPT from scratch. Then understand what it learned.

This is a research fork of Andrej Karpathy's nanochat extended with Sparse Autoencoder (SAE) based interpretability tools. You get the full nanochat training pipeline PLUS the ability to peer inside your model and discover the features it learned.

What's This About?
nanochat teaches you to build a ChatGPT-like model for ~$100.
nanochat-SAE teaches you to understand what that model actually learned.

Using Sparse Autoencoders, you can:

🔍 Discover interpretable features - Find "negation neurons", "math neurons", "sentiment neurons"
📊 Visualize activations - See which concepts light up during inference
🎛️ Steer behavior - Amplify or suppress specific features to change model outputs
🧪 Debug learning - Understand why your model succeeds or fails on tasks
🌐 Share discoveries - Export to Neuronpedia for community analysis
Why a Separate Repo?
Philosophy: Karpathy's nanochat is intentionally minimal (~8,000 lines) for educational clarity. SAE interpretability adds ~4,000 lines of advanced tooling. Rather than compromise nanochat's minimalism, we maintain this as a full-featured research branch for those who want to go deeper.

What you get here:

✅ Complete nanochat codebase (train your own LLM)
✅ SAE training pipeline (TopK, ReLU, Gated architectures)
✅ Activation collection via PyTorch hooks
✅ Feature visualization and dashboards
✅ Runtime interpretation and steering
✅ Neuronpedia integration
✅ Comprehensive documentation
Quick Start
1. Train Your Nanochat Model
This repo includes the full nanochat training pipeline:

# Clone this repo
git clone https://github.com/[YourUsername]/nanochat-sae
cd nanochat-sae

# Run the nanochat speedrun (trains a model in ~4 hours on 8xH100)
bash speedrun.sh
Your trained model checkpoint will be at models/d20/base_final.pt.

Already have a nanochat model? Just point the SAE scripts at your checkpoint.

2. Train Sparse Autoencoders
Train SAEs to decompose your model's learned features:

# Train SAE on layer 10 of your d20 model
python -m scripts.sae_train \
    --checkpoint models/d20/base_final.pt \
    --layer 10 \
    --expansion_factor 8 \
    --activation topk \
    --k 64 \
    --num_activations 1000000
This collects 1M activations from layer 10 and trains a TopK SAE with 8x expansion (10,240 features for a d20 model with 1,280 hidden dims).

Training time: ~2-4 hours on a single A100.

3. Evaluate SAE Quality
Check how well your SAE captures the model's representations:

python -m scripts.sae_eval \
    --sae_path sae_outputs/layer_10/best_model.pt \
    --generate_dashboards \
    --top_k 20
Key metrics:

Reconstruction MSE: How accurately the SAE reconstructs activations
L0 sparsity: Average number of active features (should be close to k)
Explained variance: Fraction of activation variance captured
Dead latents: Percentage of features that never activate
4. Visualize Features
Generate interactive dashboards to explore what features your model learned:

python -m scripts.sae_viz \
    --sae_path sae_outputs/layer_10/best_model.pt \
    --all_features \
    --top_k 50 \
    --output_dir feature_explorer
Open feature_explorer/index.html in your browser to see:

Top activating features
Activation frequencies
Example inputs that trigger each feature
Feature statistics
5. Runtime Interpretation (Advanced)
Track feature activations during inference:

from nanochat.gpt import GPT
from sae.runtime import InterpretableModel, load_saes

# Load your trained model
model = GPT.from_pretrained("models/d20/base_final.pt")

# Load trained SAEs
saes = load_saes("sae_outputs/")

# Wrap with interpretability
interp_model = InterpretableModel(model, saes)

# Track features during generation
with interp_model.interpretation_enabled():
    output = interp_model(input_ids)
    features = interp_model.get_active_features()
    
    # See which features fired in layer 10
    layer_10_features = features["blocks.10.hook_resid_post"]
    print(f"Active features: {(layer_10_features > 0).sum()} / {layer_10_features.shape[1]}")
What Can You Discover?
Example findings from SAE interpretability on small language models:

🚫 Negation features: Activate on "not", "never", "isn't"
🔢 Numerical features: Fire on digits, math operations
😊 Sentiment features: Distinguish positive/negative language
🌍 Entity features: Activate on proper nouns, locations
📚 Syntax features: Capture grammatical structures
Feature steering example:

# Amplify a "politeness" feature
polite_output = interp_model.steer(
    input_ids,
    feature_id=("blocks.15.hook_resid_post", 4232),
    strength=2.0  # 2x amplification
)
Repository Structure
nanochat-sae/
├── README.md                    # This file
├── speedrun.sh                  # Train nanochat model (original)
├── nanochat/                    # Core nanochat implementation
├── scripts/
│   ├── base_train.py           # Nanochat pretraining
│   ├── mid_train.py            # Nanochat midtraining
│   ├── chat_sft.py             # Nanochat supervised fine-tuning
│   ├── sae_train.py            # 🆕 Train SAEs on activations
│   ├── sae_eval.py             # 🆕 Evaluate SAE quality
│   └── sae_viz.py              # 🆕 Visualize features
├── sae/                         # 🆕 SAE implementation
│   ├── config.py               # SAE configuration
│   ├── models.py               # TopK, ReLU, Gated SAEs
│   ├── hooks.py                # Activation collection
│   ├── trainer.py              # SAE training loop
│   ├── runtime.py              # Real-time interpretation
│   ├── evaluator.py            # Evaluation metrics
│   ├── feature_viz.py          # Visualization tools
│   └── neuronpedia.py          # Neuronpedia integration
├── tests/
│   └── test_sae.py             # 🆕 SAE implementation tests
└── examples/                    # 🚧 Coming soon!
    └── tutorials/               # Step-by-step guides
Learning Path
For Beginners
Train nanochat first - Follow the main nanochat tutorial to understand the base model
Read SAE basics - Understand what Sparse Autoencoders do (Anthropic's explainer)
Run simple example - Train a single SAE on one layer
Explore features - Use visualization tools to see what your model learned
For Researchers
Multi-layer analysis - Train SAEs on multiple layers, compare features
Feature steering - Modify model behavior by intervening on features
Scaling studies - Compare features across d20, d26, d30 models
Circuit discovery - Find chains of features that implement capabilities
Publish findings - Share discoveries via Neuronpedia or papers
For Developers
Integration - Add SAE hooks to your own training loop
Custom architectures - Extend with new SAE variants
Production tools - Build monitoring dashboards for deployed models
Contribute - Submit PRs for new features and improvements
Tutorials & Examples (Coming Soon!)
We're working on comprehensive tutorials:

📘 Basic Tutorial: Train your first SAE
📗 Feature Analysis: Discover interpretable concepts
📙 Feature Steering: Modify model behavior
📕 Multi-Layer Analysis: Compare features across depths
📓 Neuronpedia Integration: Share your discoveries
📔 Case Studies: Real findings from nanochat models
Want to contribute a tutorial? Open an issue or PR!

SAE Architecture Options
TopK SAE (Recommended)
Direct sparsity control: Choose exactly k active features
Fewer dead latents: More stable training at scale
Best for: Initial exploration, interpretability research
Reference: OpenAI's scaling work
ReLU SAE
Traditional approach: ReLU activation + L1 penalty
Requires tuning: Must find good L1 coefficient
Best for: Understanding SAE fundamentals
Gated SAE
Separates magnitude and selection: More expressive
More complex: Harder to train and interpret
Best for: Advanced experiments
Performance Characteristics
Memory Usage
Activation collection: ~10-20GB per layer for 10M activations
SAE training: Requires 40GB+ VRAM for large SAEs
Runtime inference: +10GB memory for all SAEs loaded
Computational Overhead
Activation collection: <5% slowdown during training
SAE inference: 5-10% latency increase
SAE training: 2-4 hours per layer on A100
Tips for Optimization
Store activations on CPU during collection to save GPU memory
Train SAEs on subset of layers (e.g., every 5th layer)
Use smaller expansion factors (4x instead of 16x) for faster training
Enable lazy loading of SAEs to reduce memory usage
Evaluation Metrics
SAEs are evaluated on three key dimensions:

1. Reconstruction Quality
MSE Loss: Mean squared error between original and reconstructed activations
Explained Variance: Fraction of activation variance captured
Reconstruction Score: 1 - MSE/variance
2. Sparsity
L0: Average number of active features per activation
L1: Average L1 norm of feature activations
Dead Latents: Fraction of features that never activate
3. Interpretability
Activation Frequency: How often each feature fires
Top Activating Examples: Inputs that maximally activate features
Feature Descriptions: Auto-generated via Neuronpedia (optional)
Contributing
We welcome contributions! Areas for improvement:

🔬 Research: Novel SAE architectures, evaluation metrics
🎨 Visualization: Better dashboards, interactive tools
📚 Documentation: Tutorials, case studies, explanations
🔧 Engineering: Performance optimizations, bug fixes
🧪 Experiments: Discover interesting features, share findings
Getting started:

Open an issue to discuss your idea
Fork the repo and create a feature branch
Submit a PR with clear description and tests
We'll review and provide feedback
Citation
If you use nanochat-SAE in your research:

@software{nanochat_sae_2025,
  title = {nanochat-SAE: Mechanistic Interpretability for Nanochat},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[YourUsername]/nanochat-sae},
  note = {Research extension of nanochat by Andrej Karpathy}
}

@software{nanochat_2025,
  title = {nanochat: The best ChatGPT that \$100 can buy},
  author = {Karpathy, Andrej},
  year = {2025},
  url = {https://github.com/karpathy/nanochat}
}
References & Resources
Core Papers
Scaling and Evaluating Sparse Autoencoders (OpenAI, 2024)
Towards Monosemanticity (Anthropic, 2023)
Sparse Autoencoders Find Highly Interpretable Features (DeepMind, 2024)
Tools & Platforms
SAELens - Comprehensive SAE training library
Neuronpedia - Platform for sharing and exploring features
TransformerLens - Mechanistic interpretability toolkit
Community
Nanochat Discussions - Main nanochat community
Alignment Forum - Interpretability research discussions
EleutherAI Discord - AI research community
Acknowledgments
Andrej Karpathy for creating nanochat and inspiring accessible AI education
OpenAI Superalignment Team for pioneering SAE scaling research
Anthropic for mechanistic interpretability foundations
SAELens contributors for open-source SAE tools
Neuronpedia team for feature sharing infrastructure
License
MIT License (same as nanochat)

Get Started Now
# Clone and train your model
git clone https://github.com/[YourUsername]/nanochat-sae
cd nanochat-sae
bash speedrun.sh

# Train SAEs and explore
python -m scripts.sae_train --checkpoint models/d20/base_final.pt --layer 10
python -m scripts.sae_viz --sae_path sae_outputs/layer_10/best_model.pt --all_features
Questions? Open an issue or start a discussion!

Found something cool? Tweet at @karpathy and share your discoveries!

nanochat-SAE: Because understanding your $100 ChatGPT is just as important as building it.

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat-SAE,
  author = {Caleb DeLeeuw},
  title = {nanochat-SAE: Train your own ChatGPT from scratch. Then understand what it learned.},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SolshineCode/nanochat-SAE}
}
```


## License

MIT
