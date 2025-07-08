import copy

from pandas.core.computation.expressions import evaluate

from agent_handler import AgentHandler

class HierarchyManager:
    def __init__(self, status, agent_handler: AgentHandler):
        self.hierarchy = {}
        self.hierarchy_keys = list(self.hierarchy.keys())
        self.build_hierarchy(status, agent_handler)
        self.max_low_level_steps = 50

    def restruct_current_obs(self, observations_):
        for key in self.hierarchy_keys:
            self.hierarchy[key].restruct_current_obs(observations_)

    # def sim_step(self, engine):
    #     self.hierarchy["low"].sim_step(engine)

    def train(self, status, engine):

        if "high" in self.hierarchy_keys:
            # Do an Exploration Step for HighLevel Agent
            # if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["high"].env_reset or status["env_reset"]:
            #     self.hierarchy["medium"].step_without_network(status, engine)
                # Dont set "status["env_reset"] = False" just yet

            # Check if we need to do a Training Step for HighLevel Agent
            if (self.hierarchy["high"].hierarchy_steps % self.max_low_level_steps == 0
                    or self.hierarchy["high"].env_reset or status["env_reset"]):
                # Do a Training Step for HighLevel Agent
                self.hierarchy["high"].train_loop(status, engine)
                self.hierarchy["high"].hierarchy_steps = 0
                self.hierarchy["high"].hierarchy_early_stop = False

            # Always step through all low level agents (ExploreFlyAgent)
            # self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["low"].eval_loop(status, engine, evaluate=False)
            # Always step through all low level agents (PlanFlyAgent)
            _, _ = self.hierarchy["plan_low"].eval_loop(status, engine, evaluate=False)
            self.hierarchy["high"].hierarchy_steps += 1
            status["env_reset"] = False # This is needed because the User can reset the environment at any time !
        elif "medium" in self.hierarchy_keys:
            # Train Loop for MediumLevel Agent TODO Currently not implemented, will only do Exploration and NO Training
            if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["medium"].env_reset or status["env_reset"]:
                self.hierarchy["medium"].step_without_network(status, engine)
                status["env_reset"] = False # This is needed because the User can reset the environment at any time !
                self.hierarchy["medium"].hierarchy_steps = 0
                self.hierarchy["medium"].hierarchy_early_stop = False
            self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["low"].eval_loop(status, engine, evaluate=False)
            self.hierarchy["medium"].hierarchy_steps += 1
        else:
            # Train Loop for low level agent
            self.hierarchy["low"].train_loop(status, engine)

    def eval(self, status, engine):
        if "high" in self.hierarchy_keys:
            if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["medium"].env_reset or status["env_reset"]:
                self.hierarchy["medium"].step_without_network(status, engine)

            if (self.hierarchy["high"].hierarchy_steps % self.max_low_level_steps == 0
                    or self.hierarchy["high"].env_reset or status["env_reset"]):
                # Do a Training Step for HighLevel Agent
                self.hierarchy["high"].eval_loop(status, engine, evaluate=True)
                self.hierarchy["high"].hierarchy_steps = 0
                self.hierarchy["high"].hierarchy_early_stop = False
            # Always step through all low level agents (ExploreFlyAgent)
            self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["low"].eval_loop(status, engine, evaluate=False)
            # Always step through all low level agents (PlanFlyAgent)
            _, _ = self.hierarchy["plan_low"].eval_loop(status, engine, evaluate=False)
            self.hierarchy["high"].hierarchy_steps += 1
            status["env_reset"] = False # This is needed because the User can reset the environment at any time !
        if "medium" in self.hierarchy_keys:
            if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["medium"].env_reset or status["env_reset"]:
                self.hierarchy["medium"].step_without_network(status, engine)
                status["env_reset"] = False # This is needed because the User can reset the environment at any time !
                self.hierarchy["medium"].hierarchy_steps = 0
                self.hierarchy["medium"].hierarchy_early_stop = False
            self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["low"].eval_loop(status, engine, evaluate=False)
            self.hierarchy["medium"].hierarchy_steps += 1
        else:
            self.hierarchy["low"].eval_loop(status, engine, evaluate=True)


    def update_status(self, status):
        if "high" in self.hierarchy_keys:
            self.hierarchy["high"].update_status(status)
        elif "medium" in self.hierarchy_keys:
            self.hierarchy["medium"].update_status(status)
        else:
            self.hierarchy["low"].update_status(status)

    def build_hierarchy(self, status, agent_handler: AgentHandler):
        self.hierarchy[agent_handler.hierarchy_level] = agent_handler
        construct_medium = agent_handler.hierarchy_level == "medium" or agent_handler.hierarchy_level == "high"
        algorithm_name = agent_handler.original_algo
        # Construct a low level agent if the current agent is a medium level agent
        if agent_handler.hierarchy_level == "high":
            vision_range = agent_handler.algorithm.vision_range
            map_size = agent_handler.algorithm.map_size
            time_steps = status["flyAgentTimesteps"]
            status_ = copy.copy(status)
            status_["hierarchy_type"] = "PlannerFlyAgent"
            status_["model_name"] = "my_model_latest.pt"
            status_["rl_mode"] = "eval"
            status_["model_path"] = "/home/nex/Dokumente/Code/ROSHAN/models/new_solved/"
            low_level_agent_plan = AgentHandler(status_, algorithm=algorithm_name, vision_range=vision_range, map_size=map_size, time_steps=time_steps, logdir='./logs')
            low_level_agent_plan.hierarchy_level = "plan_low"
            status["console"] += "Hierarchy: High Level > Medium Level > Low Level\n"
            status["console"] += "Loading PlannerFlyAgents...\n"
            low_level_agent_plan.load_model(status_)
            self.hierarchy["plan_low"] = low_level_agent_plan
        if construct_medium:
            if agent_handler.hierarchy_level == "high":
                # Construct a medium level agent if the current agent is a high level agent
                status_ = copy.copy(status)
                status_["hierarchy_type"] = "ExploreAgent"
                status_["rl_mode"] = "eval"
                status_["console"] = ""
                medium_level_agent = AgentHandler(status_, algorithm=algorithm_name, vision_range=agent_handler.algorithm.vision_range, map_size=agent_handler.algorithm.map_size, time_steps=status["exploreAgentTimesteps"], logdir='./logs')
                self.hierarchy["medium"] = medium_level_agent
            vision_range = agent_handler.algorithm.vision_range
            map_size = agent_handler.algorithm.map_size
            status_ = copy.copy(status)
            time_steps = status_["flyAgentTimesteps"]
            status_["hierarchy_type"] = "ExploreFlyAgent"
            status_["model_name"] = "my_model_latest.pt"
            status_["model_path"] = "/home/nex/Dokumente/Code/ROSHAN/models/new_solved/"
            status_["rl_mode"] = "eval"
            status_["console"] = ""
            low_level_agent = AgentHandler(status_, algorithm=algorithm_name, vision_range=vision_range, map_size=map_size, time_steps=time_steps, logdir='./logs')
            if agent_handler.hierarchy_level != "high": status["console"] += "Hierarchy: Medium Level & Low Level\n"
            status["console"] += "Loading ExploreFlyAgent Model...\n"
            low_level_agent.load_model(status_)
            self.hierarchy["low"] = low_level_agent
        else:
            status["console"] += "Hierarchy: Low Level\n"

        self.hierarchy_keys = list(self.hierarchy.keys())
