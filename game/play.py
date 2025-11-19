"""
Main entry point for playing the SAE Guardian game.

Usage:
    python -m game.play --scenario emergency_shutdown --max-steps 50
    python -m game.play --scenario model_theft --interactive
"""

import argparse
import sys
from pathlib import Path
import torch

from game.core import Game, GameState
from game.core.agent import create_agent
from game.core.monitor import SAEMonitor
from game.scenarios import get_scenario, SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser(description="Play SAE Guardian game")

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/d20/base_final.pt',
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--sae-dir',
        type=str,
        default='sae_outputs',
        help='Directory containing SAE checkpoints'
    )

    parser.add_argument(
        '--scenario',
        type=str,
        default='emergency_shutdown',
        choices=list(SCENARIOS.keys()),
        help='Scenario to play'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum number of steps'
    )

    parser.add_argument(
        '--intervention-budget',
        type=int,
        default=5,
        help='Number of interventions allowed'
    )

    parser.add_argument(
        '--monitoring-budget',
        type=int,
        default=50,
        help='Number of features that can be actively monitored'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('game_results'),
        help='Directory to save results'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive mode (player makes decisions)'
    )

    parser.add_argument(
        '--auto-suggest-watchers',
        action='store_true',
        help='Automatically suggest features to watch'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )

    return parser.parse_args()


def interactive_callback(monitor: SAEMonitor):
    """
    Interactive callback for player decisions.

    Returns a function that asks player whether to intervene.
    """
    def callback(monitoring_report):
        print("\n" + "="*60)
        print("PLAYER DECISION REQUIRED")
        print("="*60)
        print(f"Suspicion Score: {monitoring_report.suspicion_score:.2f}")
        print(f"New Alerts: {len(monitoring_report.new_alerts)}")
        print(f"Anomalies Detected: {len(monitoring_report.anomalies)}")

        if monitoring_report.new_alerts:
            print("\nRecent Alerts:")
            for alert in monitoring_report.new_alerts[-3:]:
                print(f"  [{alert.severity.upper()}] {alert.message}")

        print(f"\nRemaining Interventions: {monitor.intervention_budget}")

        response = input("\nIntervene? (y/n): ").strip().lower()
        return response == 'y'

    return callback


def setup_watchers(monitor: SAEMonitor, auto_suggest: bool = False):
    """
    Set up feature watchers.

    In auto mode, suggests watchers based on initial activity.
    In manual mode, player can specify watchers.
    """
    if auto_suggest:
        print("\nAnalyzing features to suggest watchers...")
        print("(This will be available after a few steps)")
        return

    print("\nWatcher setup skipped (can be done during gameplay)")


def main():
    args = parse_args()

    print("="*60)
    print("SAE GUARDIAN: LLM Exfiltration Detection Game")
    print("="*60)
    print()

    # Check if checkpoint and SAE directory exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train a model first or specify correct path")
        return 1

    if not Path(args.sae_dir).exists():
        print(f"Error: SAE directory not found: {args.sae_dir}")
        print("Please train SAEs first or specify correct path")
        return 1

    # Get scenario
    scenario = get_scenario(args.scenario)
    print(f"Scenario: {scenario.config.name}")
    print(f"Difficulty: {scenario.config.difficulty}")
    print(f"Description: {scenario.config.description}")
    print()

    # Show hints
    print("HINTS FOR PLAYERS:")
    for i, hint in enumerate(scenario.get_hints(), 1):
        print(f"{i}. {hint}")
    print()

    # Create agent and environment
    try:
        agent, environment = create_agent(
            checkpoint_path=args.checkpoint,
            sae_dir=args.sae_dir,
            scenario=args.scenario,
            device=args.device
        )
    except Exception as e:
        print(f"Error creating agent: {e}")
        return 1

    # Create monitor
    monitor = SAEMonitor(
        saes=agent.interp_model.saes,
        intervention_budget=args.intervention_budget,
        monitoring_budget=args.monitoring_budget,
        device=args.device
    )

    # Setup watchers
    setup_watchers(monitor, auto_suggest=args.auto_suggest_watchers)

    # Create game
    player_callback = None
    if args.interactive:
        player_callback = interactive_callback(monitor)

    game = Game(
        agent=agent,
        monitor=monitor,
        environment=environment,
        max_steps=args.max_steps,
        player_callback=player_callback
    )

    # Run game
    print("Starting game...")
    print()

    try:
        result = game.run(verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nError during game: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"game_{int(result.timestamp)}.json"
    result.save(result_path)

    print(f"\nResults saved to: {result_path}")

    # Print monitoring statistics
    print("\n" + "="*60)
    print("MONITORING STATISTICS")
    print("="*60)
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Print environment statistics
    print("\n" + "="*60)
    print("ENVIRONMENT STATISTICS")
    print("="*60)
    env_stats = environment.get_statistics()
    for key, value in env_stats.items():
        print(f"{key}: {value}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
