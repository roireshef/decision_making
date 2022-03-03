import inspect
import pkgutil
from importlib import import_module
from pathlib import Path
from typing import Callable, Any, Dict

from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print


class RLRegistry:
    registered_models = {}
    registered_action_space_adapters = {}
    registered_reward_functions = {}

    @staticmethod
    def init():
        RLRegistry.registered_models = RLRegistry.register_models()
        RLRegistry.registered_action_space_adapters = RLRegistry.register_action_space_adapters()
        RLRegistry.registered_reward_functions = RLRegistry.register_reward_functions()

    @staticmethod
    def register_models() -> Dict[str, Any]:
        """ Registers in ray.tune.registry all TorchModel implementations
        under decision_making.src.rl_agent.agents.models"""
        from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
        import decision_making.src.rl_agent.agents.models
        return RLRegistry.register(ModelCatalog.register_custom_model, TorchModelV2,
                                   decision_making.src.rl_agent.agents.models)

    @staticmethod
    def register_action_space_adapters() -> Dict[str, Any]:
        """ Registers in ray.tune.registry all ActionSpaceAdapter implementations
        under decision_making.src.rl_agent.environments.action_space
        This function uses ModelCatalog.register_custom_preprocessor to register a non "preprocessor" objects
         since ray's registry only deals with known categories for its registered items. for more details,
         see: ray.tune.registry.py """
        from decision_making.src.rl_agent.environments.action_space.action_space_adapter import ActionSpaceAdapter
        import decision_making.src.rl_agent.environments.action_space
        return RLRegistry.register(ModelCatalog.register_custom_preprocessor, ActionSpaceAdapter,
                                   decision_making.src.rl_agent.environments.action_space)

    @staticmethod
    def register(reg_func: Callable[[str, Any], None], parent_class: Any, package: Any) -> Dict[str, Any]:
        """
        Takes a registry callable <reg_func>, and calls this function for each descendant of
        <parent_class> that is located under <package>
        :param reg_func: function that takes (name, object) as inputs and registers <object> by the name <name>
        :param parent_class: parent class to look for its descendants
        :param package: package to look for descendant in
        :return: a dictionary of IDs of registered object names
        """
        items_found = {}

        # iterate over all files in a package
        for _, name, is_pkg in pkgutil.iter_modules([Path(package.__file__).parent]):
            # import every file that isn't itself a package
            if not is_pkg:
                imported_module = import_module('.' + name, package=package.__name__)

                # look for all classes in that file
                for name, cls in inspect.getmembers(imported_module, lambda obj: inspect.isclass(obj)):
                    # if classes match the pattern we look for, register it and return its name
                    if issubclass(cls, parent_class) and cls.__name__ != parent_class.__name__:
                        reg_func(cls.__name__, cls)         # register class
                        items_found[cls.__name__] = cls     # add item to return list

        return items_found


if __name__ == "__main__":
    RLRegistry.init()

    print("\nRLRegistry successfully initialized! Registered items:\n\n%s" %
          pretty_print({"MODELS": RLRegistry.registered_models,
                        "ACTION_SPACE_ADAPTERS": RLRegistry.registered_action_space_adapters}))
