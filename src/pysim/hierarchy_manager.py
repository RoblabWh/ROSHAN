import copy

from pandas.core.computation.expressions import evaluate

from agent_handler import AgentHandler

class HierarchyManager:
    def __init__(self, status, agent_handler: AgentHandler):
        self.hierarchy = {}
        self.hierarchy_keys = list(self.hierarchy.keys())
        self.build_hierarchy(status, agent_handler)
        self.max_low_level_steps = 750

    def restruct_current_obs(self, observations_):
        for key in self.hierarchy_keys:
            self.hierarchy[key].restruct_current_obs(observations_)

    def sim_step(self, engine):
        self.hierarchy["low"].sim_step(engine)

    def train(self, status, engine):

        if "medium" in self.hierarchy_keys:
            # Train Loop for medium level agent
            if self.hierarchy["medium"].hierarchy_steps % self.max_low_level_steps == 0 or \
                self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["medium"].env_reset or status["env_reset"]:
                self.hierarchy["medium"].train_loop(status, engine)
                status["env_reset"] = False # This is needed because the User can reset the environment at any time !
                self.hierarchy["medium"].hierarchy_steps = 0
                self.hierarchy["medium"].hierarchy_early_stop = False
            else:
                self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["low"].eval_loop(status, engine, evaluate=False)
            self.hierarchy["medium"].hierarchy_steps += 1
        else:
            # Train Loop for low level agent
            self.hierarchy["low"].train_loop(status, engine)

    def eval(self, status, engine):
        if "medium" in self.hierarchy_keys:
            self.hierarchy["medium"].eval_loop(status, engine, evaluate=True)
        else:
            self.hierarchy["low"].eval_loop(status, engine, evaluate=True)

    def restructure_data(self, observations_):
        if "medium" in self.hierarchy_keys:
            return self.hierarchy["medium"].restructure_data(observations_)
        else:
            return self.hierarchy["low"].restructure_data(observations_)

    def update_status(self, status):
        if "medium" in self.hierarchy_keys:
            self.hierarchy["medium"].update_status(status)
        else:
            self.hierarchy["low"].update_status(status)

    def build_hierarchy(self, status, agent_handler: AgentHandler):
        self.hierarchy[agent_handler.hierarchy_level] = agent_handler

        # Construct a low level agent if the current agent is a medium level agent
        if agent_handler.hierarchy_level == "medium":
            vision_range = agent_handler.algorithm.vision_range
            map_size = agent_handler.algorithm.map_size
            status_ = copy.copy(status)
            time_steps = status_["flyAgentTimesteps"]
            status_["hierarchy_type"] = "FlyAgent"
            status_["model_name"] = "my_model_obj_v1.pt"
            status_["model_path"] = "/home/nex/Dokumente/Code/ROSHAN/models/new_solved/"
            status_["rl_mode"] = "eval"
            status_["console"] = ""
            low_level_agent = AgentHandler(status_, algorithm='PPO', vision_range=vision_range, map_size=map_size, time_steps=time_steps, logdir='./logs')
            status["console"] += "Hierarchy: Medium Level & Low Level\n"
            status["console"] += "Loading Low Level Model...\n"
            low_level_agent.load_model(status_)
            self.hierarchy["low"] = low_level_agent
        else:
            status["console"] += "Hierarchy: Low Level\n"

        self.hierarchy_keys = list(self.hierarchy.keys())
