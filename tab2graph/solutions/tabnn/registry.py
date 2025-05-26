__all__ = ['tabnn', 'get_tabnn_class']

_TABNN_REGISTRY = {}

def tabnn(tabnn_class):
    global _TABNN_REGISTRY
    _TABNN_REGISTRY[tabnn_class.name] = tabnn_class
    return tabnn_class

def get_tabnn_class(name : str):
    global _TABNN_REGISTRY
    tabnn_class = _TABNN_REGISTRY.get(name, None)
    if tabnn_class is None:
        raise ValueError(f"Cannot find the TabNN class of name {name}.")
    return tabnn_class
