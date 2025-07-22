import copy
from agent_handler import AgentHandler

class HierarchyManager:
    def __init__(self, status, config, agent_handler: AgentHandler):
        self.config = config
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

        # Construct a low level agent if the current agent is a medium level agent
        if agent_handler.hierarchy_level == "high":
            low_level_agent_plan = AgentHandler(config=self.config, agent_type="fly_agent", subtype="PlannerFlyAgent", mode="eval", logdir='./logs')
            low_level_agent_plan.hierarchy_level = "plan_low"
            status["console"] += "Hierarchy: High Level > Medium Level > Low Level\n"
            status["console"] += "Loading PlannerFlyAgents...\n"
            _, _ = low_level_agent_plan.load_model()
            self.hierarchy["plan_low"] = low_level_agent_plan
        if construct_medium:
            # Construct a medium level agent if the current agent is a high level agent
            if agent_handler.hierarchy_level == "high":
                medium_level_agent = AgentHandler(config=self.config, agent_type="explore_agent", mode="eval", logdir='./logs')
                self.hierarchy["medium"] = medium_level_agent
            low_level_agent = AgentHandler(config=self.config, agent_type="fly_agent", subtype="ExploreFlyAgent", mode="eval", logdir='./logs')
            if agent_handler.hierarchy_level != "high": status["console"] += "Hierarchy: Medium Level & Low Level\n"
            status["console"] += "Loading ExploreFlyAgent Model...\n"
            _, _ = low_level_agent.load_model()
            self.hierarchy["low"] = low_level_agent
        else:
            status["console"] += "Hierarchy: Low Level\n"

        self.hierarchy_keys = list(self.hierarchy.keys())
