#!/usr/bin/env python3
"""
Script to visualize GRPO training outputs from TRL suffix learner.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_jsonl(file_path):
    """Load JSONL file and return list of records."""
    records = []
    if file_path.exists():
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def visualize_experience_history(grpo_outputs_dir):
    """Create visualizations for experience history and training progress."""
    grpo_outputs_dir = Path(grpo_outputs_dir)

    # Load data
    experience_file = grpo_outputs_dir / "experience_history.jsonl"
    training_file = grpo_outputs_dir / "training_sessions.jsonl"
    generation_file = grpo_outputs_dir / "grpo_generations.jsonl"
    metrics_file = grpo_outputs_dir / "training_metrics.jsonl"

    experiences = load_jsonl(experience_file)
    trainings = load_jsonl(training_file)
    generations = load_jsonl(generation_file)
    metrics = load_jsonl(metrics_file)

    print(f"Loaded {len(experiences)} experiences")
    print(f"Loaded {len(trainings)} training sessions")
    print(f"Loaded {len(generations)} generation batches")
    print(f"Loaded {len(metrics)} metrics entries")

    if not experiences:
        print("No experience data found!")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("GRPO Training Analysis", fontsize=16)

    # 1. Reward distribution over time
    ax = axes[0, 0]
    rewards = [exp["reward"] for exp in experiences]
    timestamps = [
        datetime.fromisoformat(exp["timestamp"]) for exp in experiences
    ]
    ax.scatter(timestamps, rewards, alpha=0.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards Over Time")
    ax.tick_params(axis="x", rotation=45)

    # 2. Experience accumulation over time
    ax = axes[0, 1]
    if experiences:
        exp_ids = [
            exp.get("experience_id", i) for i, exp in enumerate(experiences)
        ]
        exp_rewards = [exp["reward"] for exp in experiences]

        # Plot cumulative average reward
        cumulative_avg = np.cumsum(exp_rewards) / np.arange(
            1, len(exp_rewards) + 1
        )
        ax.plot(exp_ids, cumulative_avg, "b-", label="Cumulative Avg Reward")

        # Plot rolling average (window of 10)
        if len(exp_rewards) > 10:
            rolling_avg = pd.Series(exp_rewards).rolling(window=10).mean()
            ax.plot(
                exp_ids, rolling_avg, "r-", alpha=0.7, label="Rolling Avg (10)"
            )

        ax.set_xlabel("Experience ID")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Trends Over Experience Collection")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Training session statistics
    if trainings:
        ax = axes[1, 0]
        session_ids = [t["session_id"] for t in trainings]
        num_experiences = [t.get("num_experiences", 0) for t in trainings]
        num_unique_prompts = [
            t.get("num_unique_prompts", t.get("num_prompts", 0))
            for t in trainings
        ]
        global_avg_rewards = [t.get("global_avg_reward", 0) for t in trainings]

        ax2 = ax.twinx()
        ax.bar(
            session_ids, num_experiences, alpha=0.7, label="Num Experiences"
        )
        ax.bar(
            session_ids, num_unique_prompts, alpha=0.5, label="Unique Prompts"
        )
        ax2.plot(session_ids, global_avg_rewards, "r-o", label="Avg Reward")

        ax.set_xlabel("Session ID")
        ax.set_ylabel("Count")
        ax2.set_ylabel("Average Reward")
        ax.set_title("Training Sessions Overview")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    # 4. Generation quality after training
    if generations:
        ax = axes[1, 1]
        session_rewards = {}
        for gen in generations:
            session_id = gen["session_id"]
            if session_id not in session_rewards:
                session_rewards[session_id] = []
            session_rewards[session_id].extend(gen["rewards"])

        sessions = sorted(session_rewards.keys())
        avg_gen_rewards = [
            sum(session_rewards[s]) / len(session_rewards[s]) for s in sessions
        ]
        max_gen_rewards = [max(session_rewards[s]) for s in sessions]

        ax.plot(sessions, avg_gen_rewards, "b-o", label="Avg Reward")
        ax.plot(sessions, max_gen_rewards, "g-^", label="Max Reward")
        ax.set_xlabel("Session ID")
        ax.set_ylabel("Reward")
        ax.set_title("Generation Quality by Training Session")
        ax.legend()

    plt.tight_layout()

    # Save figure
    output_file = grpo_outputs_dir / "grpo_analysis.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved analysis plot to: {output_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total experiences collected: {len(experiences)}")

    if experiences:
        # Calculate unique prompts and average rewards
        unique_prompts = set(exp["prompt"] for exp in experiences)
        all_rewards = [exp["reward"] for exp in experiences]
        print(f"Unique prompts: {len(unique_prompts)}")
        print(
            f"Average reward across all experiences: {sum(all_rewards) / len(all_rewards):.3f}"
        )
        print(f"Max reward achieved: {max(all_rewards):.3f}")
        print(f"Min reward achieved: {min(all_rewards):.3f}")

    if trainings:
        print(f"\nTotal training sessions: {len(trainings)}")
        # Update to use correct field names
        total_experiences = sum(
            t.get("num_experiences", t.get("num_prompts", 0))
            for t in trainings
        )
        print(f"Total experiences used in training: {total_experiences}")

        # Show average rewards from training sessions
        if any("global_avg_reward" in t for t in trainings):
            print("\n=== Training Session Average Rewards ===")
            for t in trainings[-5:]:  # Show last 5 sessions
                if "global_avg_reward" in t:
                    print(
                        f"  Session {t['session_id']}: {t['global_avg_reward']:.3f}"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GRPO training outputs"
    )
    parser.add_argument(
        "grpo_outputs_dir",
        type=str,
        help="Path to grpo_training_outputs directory",
    )
    args = parser.parse_args()

    visualize_experience_history(args.grpo_outputs_dir)


if __name__ == "__main__":
    main()
