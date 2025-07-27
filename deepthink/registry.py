from deepthink.agent import Agent

AGENT_REGISTRY = {}


def register(name):
    def decorator(item):
        key = name if name is not None else item.__name__
        AGENT_REGISTRY[key] = item
        return item

    return decorator


def get_agent_cls(name) -> Agent:
    return AGENT_REGISTRY.get(name)


def list_agents() -> list[str]:
    return list(AGENT_REGISTRY.keys())
