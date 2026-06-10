import logging
import random

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


def get_user_name_from_environment(suite_name: str, environment) -> str:
    """Extract user name from suite environment.

    Shared utility for all learners.

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
