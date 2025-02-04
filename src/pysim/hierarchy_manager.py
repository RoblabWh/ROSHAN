import copy

from pandas.core.computation.expressions import evaluate

from agent_handler import AgentHandler

class HierarchyManager:
    def __init__(self, status, agent_handler: AgentHandler):
        self.hierarchy = {}
        self.hierarchy_keys = list(self.hierarchy.keys())
        self.build_hierarchy(status, agent_handler)
        self.low_level_steps = 250

    def initial_observation(self, observations_):
        for key in self.hierarchy_keys:
            self.hierarchy[key].initial_observation(observations_)

    def train(self, status, engine, steps):

        if "medium" in self.hierarchy_keys:
            # Train Loop for medium level agent
            if steps % self.low_level_steps == 0:
                self.hierarchy["medium"].train_loop(status, engine)
            self.hierarchy["low"].eval_loop(status, engine, evaluate=False)
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
        self.hierarchy[agent_handler.hierachy_level] = agent_handler

        # Construct a low level agent if the current agent is a medium level agent
        if agent_handler.hierachy_level == "medium":
            vision_range = agent_handler.algorithm.vision_range
            time_steps = agent_handler.algorithm.time_steps
            status_ = copy.copy(status)
            status_["agent_type"] = "FlyAgent"
            status_["model_name"] = "my_model_obj_v1.pt"
            status_["model_path"] = "/home/nex/Dokumente/Code/ROSHAN/models/Solved.100/"
            status_["rl_mode"] = "eval"
            status_["console"] = ""
            low_level_agent = AgentHandler(status_, algorithm="ppo", vision_range=vision_range, time_steps=time_steps, logdir='./logs')
            status["console"] += "Hierarchy: Medium Level & Low Level\n"
            status["console"] = low_level_agent.load_model(status_)
            self.hierarchy["low"] = low_level_agent
        else:
            status["console"] += "Hierarchy: Low Level\n"

        self.hierarchy_keys = list(self.hierarchy.keys())
