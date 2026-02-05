from agentdojo.default_suites.v1.travel.injection_tasks import (  # noqa: F401 - Register tasks
    TravelInjectionTask,
)
from agentdojo.default_suites.v1.travel.task_suite import (
    task_suite as travel_task_suite,
)
from agentdojo.default_suites.v1.travel.user_tasks import (  # noqa: F401 - Register tasks
    TravelUserTask,
)

__all__ = ["travel_task_suite"]
