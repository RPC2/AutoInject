import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure logging
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def log_metrics(
    metrics,
    iteration,
    log_dir,
    current_entropy_coeff,
    mean_utility,
    total_steps,
):
    """Log training metrics to console and file."""
    log_str = f"Iteration {iteration} training metrics: "
    log_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    # Also add current annealed values
    log_str += f", current_ent_coeff={current_entropy_coeff:.4f}"
    logger.info(log_str)

    # Save metrics to file
    metrics_log_path = f"{log_dir}/metrics_log.csv"
    is_new_file = not os.path.exists(metrics_log_path)
    with open(metrics_log_path, "a") as f:
        # Combine training metrics with current state values
        log_data = metrics.copy()
        log_data["iteration"] = iteration
        log_data["current_entropy_coeff"] = current_entropy_coeff
        # Add latest utility score if available
        log_data["mean_utility"] = (
            mean_utility if mean_utility is not None else float("nan")
        )
        log_data["total_steps"] = total_steps

        # Dynamically create header based on keys in the first logged entry
        header = ",".join(sorted(log_data.keys()))
        if is_new_file or os.path.getsize(metrics_log_path) == 0:
            f.write(header + "\n")

        # Write values in the same sorted order as the header
        values = [f"{log_data[k]:.6f}" for k in sorted(log_data.keys())]
        f.write(",".join(values) + "\n")


def plot_metrics(metrics, log_dir):
    """Plot and save training metrics, including new ones."""
    if not metrics["total_loss"]:
        return

    # Increase figure size and use a 4x3 layout
    plt.figure(figsize=(24, 20))

    # Plot losses (1)
    plt.subplot(4, 3, 1)
    plt.plot(metrics["policy_loss"], label="Policy Loss")
    plt.plot(metrics["total_loss"], label="Total Loss")
    plt.title("Training Losses")
    plt.xlabel("Training Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot entropy and KL (2)
    plt.subplot(4, 3, 2)
    ax1 = plt.gca()
    color = "tab:red"
    ax1.set_xlabel("Training Batch")
    ax1.set_ylabel("Entropy", color=color)
    ax1.plot(metrics["entropy"], label="Entropy", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, axis="y")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Approx KL", color=color)
    ax2.plot(metrics["kl_divergence"], label="Approx KL", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    plt.title("Entropy & Approx KL Divergence")

    # Plot Attack Success Rate and Utility (3)
    plt.subplot(4, 3, 3)
    ax1 = plt.gca()
    color = "tab:green"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Attack Success Rate", color=color)
    ax1.plot(
        metrics["attack_success_rate"],
        label="Attack Success Rate",
        color=color,
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, axis="y")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color = "tab:purple"
    ax2.set_ylabel("Mean Utility", color=color)
    ax2.plot(metrics["mean_utility"], label="Mean Utility", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    plt.title("Attack Success Rate & Mean Utility")

    # Plot Smoothed Episode Reward (4)
    plt.subplot(4, 3, 4)
    window_size = 10
    episode_rewards = metrics["episode_reward"]
    if len(episode_rewards) >= window_size:
        smoothed_rewards = np.convolve(
            episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            np.arange(window_size - 1, len(episode_rewards)),
            smoothed_rewards,
            label=f"Smoothed (window={window_size})",
            color="tab:orange",
        )
    plt.plot(episode_rewards, label="Raw", alpha=0.3, color="tab:orange")
    plt.title("Smoothed Episode Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    # Plot Gradient Norm (5)
    plt.subplot(4, 3, 5)
    plt.plot(metrics["grad_norm"], label="Gradient Norm")
    plt.title("Gradient Norm (Pre-Clipping)")
    plt.xlabel("Training Batch")
    plt.ylabel("Norm")
    plt.legend()
    plt.grid(True)

    # Plot Annealing Parameters (6)
    plt.subplot(4, 3, 6)
    ax1 = plt.gca()
    color = "tab:orange"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Entropy Coefficient", color=color)
    ax1.plot(
        metrics["entropy_coeff"], label="Entropy Coefficient", color=color
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, axis="y")
    plt.title("Entropy Coefficient Annealing")

    # Plot Top Template Success Rates (7)
    plt.subplot(4, 3, 7)
    template_success_rates = {}
    for template_idx, success_rates in metrics["template_success"].items():
        if success_rates:  # Only include templates that have been used
            template_success_rates[template_idx] = np.mean(success_rates)

    # Sort templates by success rate and take top 10
    top_templates = sorted(
        template_success_rates.items(), key=lambda x: x[1], reverse=True
    )[:10]
    if top_templates:
        templates, rates = zip(*top_templates)
        plt.bar(range(len(templates)), rates)
        plt.xticks(range(len(templates)), templates, rotation=45, ha="right")
        plt.title("Top 10 Template Success Rates")
        plt.ylabel("Average Success Rate")
        plt.ylim(0, 1.05)
    else:
        plt.title("Template Success Rates (No data)")

    # Plot Template Usage (8)
    plt.subplot(4, 3, 8)
    template_usage = sorted(
        metrics["template_usage"].items(), key=lambda x: x[1], reverse=True
    )[:10]
    if template_usage:
        templates, counts = zip(*template_usage)
        plt.bar(range(len(templates)), counts)
        plt.xticks(range(len(templates)), templates, rotation=45, ha="right")
        plt.title("Top 10 Template Usage")
        plt.ylabel("Usage Count")
    else:
        plt.title("Template Usage (No data)")

    # Plot Token Success Rates (9)
    plt.subplot(4, 3, 9)
    token_success_rates = {}
    for token, usage in metrics["modifier_usage"].items():
        if usage > 0:  # Only include tokens that have been used
            # Get the success rates for attacks where this token was used
            token_successes = metrics.get("token_successes", {}).get(token, [])
            if (
                token_successes
            ):  # Only calculate if we have success data for this token
                token_success_rates[token] = np.mean(token_successes)
            else:
                token_success_rates[token] = 0.0

    # Sort tokens by success rate and take top 10
    top_tokens = sorted(
        token_success_rates.items(), key=lambda x: x[1], reverse=True
    )[:10]
    if top_tokens:
        tokens, rates = zip(*top_tokens)
        plt.bar(range(len(tokens)), rates)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        plt.title("Top 10 Token Success Rates")
        plt.ylabel("Average Success Rate")
        plt.ylim(0, 1.05)
    else:
        plt.title("Token Success Rates (No data)")

    # Plot Token Usage (10)
    plt.subplot(4, 3, 10)
    modifier_usage = sorted(
        metrics["modifier_usage"].items(), key=lambda x: x[1], reverse=True
    )[:10]
    if modifier_usage:
        modifiers, counts = zip(*modifier_usage)
        plt.bar(range(len(modifiers)), counts)
        plt.xticks(range(len(modifiers)), modifiers, rotation=45, ha="right")
        plt.title("Top 10 Token Usage")
        plt.ylabel("Usage Count")
    else:
        plt.title("Token Usage (No data)")

    # Plot Value Loss (11)
    plt.subplot(4, 3, 11)
    plt.plot(metrics["value_loss"], label="Value Loss")
    plt.title("Value Loss")
    plt.xlabel("Training Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Policy Loss vs Value Loss (12)
    plt.subplot(4, 3, 12)
    plt.plot(metrics["policy_loss"], label="Policy Loss")
    plt.plot(metrics["value_loss"], label="Value Loss")
    plt.title("Policy Loss vs Value Loss")
    plt.xlabel("Training Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = f"{log_dir}/metrics_detailed_v4.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved detailed metrics plot to {plot_path}")


def get_user_name_from_environment(suite_name: str, environment) -> str:
    """Extract user name from suite environment.

    Shared utility for all learners (PPO, TRL suffix, etc.)

    Args:
        suite_name: Suite name ("banking", "travel", "workspace", "slack")
        environment: From suite.load_and_inject_default_environment({})

    Returns:
        User name string, or "User" as fallback
    """
    suite_name = suite_name.lower()

    try:
        if suite_name == "banking":
            return (
                f"{environment.user_account.first_name} "
                f"{environment.user_account.last_name}"
            )
        elif suite_name == "travel":
            return (
                f"{environment.user.first_name} "
                f"{environment.user.last_name}"
            )
        elif suite_name == "workspace":
            email = environment.inbox.account_email
            name_part = email.split("@")[0]
            return name_part.replace(".", " ").title()
        elif suite_name == "slack":
            if hasattr(environment, "slack") and environment.slack.users:
                return environment.slack.users[0]
        return "User"
    except (AttributeError, IndexError, KeyError):
        return "User"
