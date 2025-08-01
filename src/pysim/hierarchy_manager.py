from agent_handler import AgentHandler
from utils import SimulationBridge
import logging


class HierarchyManager:
    def __init__(self, config, sim_bridge : SimulationBridge):
        self.config = config
        self.sim_bridge = sim_bridge
        self.hierarchy = {}
        self.hierarchy_keys = list(self.hierarchy.keys())
        self.logger = logging.getLogger("HierarchyManager")
        self.build_hierarchy()
        self.max_low_level_steps = 50

    def restruct_current_obs(self, observations_):
        for key in self.hierarchy_keys:
            self.hierarchy[key].restruct_current_obs(observations_)

    def train(self, engine):

        if "high" in self.hierarchy_keys:
            # Do an Exploration Step for HighLevel Agent
            # if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["high"].env_reset or status["env_reset"]:
            #     self.hierarchy["medium"].step_without_network(status, engine)
                # Dont set "status["env_reset"] = False" just yet
            # Check if we need to do a Training Step for HighLevel Agent
            if (self.hierarchy["high"].hierarchy_steps % self.max_low_level_steps == 0
                    or self.hierarchy["high"].env_reset or self.sim_bridge.get("env_reset")):
                # Do a Training Step for HighLevel Agent
                self.hierarchy["high"].train_loop(engine)
                self.hierarchy["high"].hierarchy_steps = 0
                self.hierarchy["high"].hierarchy_early_stop = False

            # Always step through all low level agents (ExploreFlyAgent)
            # self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["explore_low"].eval_loop(status, engine, evaluate=False)
            # Always step through all low level agents (PlanFlyAgent)
            _, _ = self.hierarchy["plan_low"].eval_loop(engine, evaluate=False)
            self.hierarchy["high"].hierarchy_steps += 1
            # This is needed because the User can reset the environment at any time !
            self.sim_bridge.set("env_reset", False)
        elif "medium" in self.hierarchy_keys:
            # Train Loop for MediumLevel Agent TODO Currently not implemented, will only do Exploration and NO Training
            if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["medium"].env_reset or self.sim_bridge.get("env_reset"):
                self.hierarchy["medium"].step_without_network(engine)
                # This is needed because the User can reset the environment at any time !
                self.sim_bridge.set("env_reset", False)
                self.hierarchy["medium"].hierarchy_steps = 0
                self.hierarchy["medium"].hierarchy_early_stop = False
            self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["explore_low"].eval_loop(engine, evaluate=False)
            self.hierarchy["medium"].hierarchy_steps += 1
        else:
            # Train Loop for low level agent
            self.hierarchy["low"].train_loop(engine)

    def eval(self, engine):
        if "high" in self.hierarchy_keys:
            if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["medium"].env_reset or self.sim_bridge.get("env_reset"):
                self.hierarchy["medium"].step_without_network(engine)

            if (self.hierarchy["high"].hierarchy_steps % self.max_low_level_steps == 0
                    or self.hierarchy["high"].env_reset or self.sim_bridge.get("env_reset")):
                # Do a Training Step for HighLevel Agent
                self.hierarchy["high"].eval_loop(engine, evaluate=True)
                self.hierarchy["high"].hierarchy_steps = 0
                self.hierarchy["high"].hierarchy_early_stop = False
            # Always step through all low level agents (ExploreFlyAgent)
            self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["explore_low"].eval_loop(engine, evaluate=False)
            # Always step through all low level agents (PlanFlyAgent)
            _, _ = self.hierarchy["plan_low"].eval_loop(engine, evaluate=False)
            self.hierarchy["high"].hierarchy_steps += 1
            # This is needed because the User can reset the environment at any time !
            self.sim_bridge.set("env_reset", False)
        if "medium" in self.hierarchy_keys:
            if self.hierarchy["medium"].hierarchy_early_stop or self.hierarchy["medium"].env_reset or self.sim_bridge.get("env_reset"):
                self.hierarchy["medium"].step_without_network(engine)
                # This is needed because the User can reset the environment at any time !
                self.sim_bridge.set("env_reset", False)
                self.hierarchy["medium"].hierarchy_steps = 0
                self.hierarchy["medium"].hierarchy_early_stop = False
            self.hierarchy["medium"].hierarchy_early_stop, _ = self.hierarchy["explore_low"].eval_loop(engine, evaluate=False)
            self.hierarchy["medium"].hierarchy_steps += 1
        else:
            self.hierarchy["low"].eval_loop(engine, evaluate=True)


    def update_status(self):
        if "high" in self.hierarchy_keys:
            self.hierarchy["high"].update_status()
        elif "medium" in self.hierarchy_keys:
            self.hierarchy["medium"].update_status()
        else:
            self.hierarchy["low"].update_status()

    def build_hierarchy(self):
        # Create the top level Agent Object
        agent_handler = AgentHandler(config=self.config,
                             agent_type=self.config["settings"]["hierarchy_type"],
                             mode=self.config["settings"]["rl_mode"],
                             sim_bridge=self.sim_bridge,
                             is_sub_agent=False)
        agent_handler.load_model(change_status=True)

        self.hierarchy[agent_handler.hierarchy_level] = agent_handler
        construct_medium = agent_handler.hierarchy_level == "medium" or agent_handler.hierarchy_level == "high"

        # Construct a low level agent if the current agent is a medium level agent
        if agent_handler.hierarchy_level == "high":
            planner_fly_agent = AgentHandler(config=self.config,
                                                sim_bridge=self.sim_bridge,
                                                agent_type="fly_agent",
                                                subtype="PlannerFlyAgent",
                                                mode="eval")
            planner_fly_agent.hierarchy_level = "plan_low"
            planner_fly_agent.load_model()
            self.hierarchy["plan_low"] = planner_fly_agent

        if construct_medium:
            # Construct a medium level agent if the current agent is a high level agent
            if agent_handler.hierarchy_level == "high":
                medium_level_agent = AgentHandler(config=self.config,
                                                  sim_bridge=self.sim_bridge,
                                                  agent_type="explore_agent",
                                                  mode="eval")
                self.hierarchy["medium"] = medium_level_agent
            explore_fly_agent = AgentHandler(config=self.config,
                                           sim_bridge=self.sim_bridge,
                                           agent_type="fly_agent",
                                           subtype="ExploreFlyAgent",
                                           mode="eval")
            explore_fly_agent.load_model()
            self.hierarchy["explore_low"] = explore_fly_agent

        highest_hierarchy_level = max(self.hierarchy.keys(), key=lambda x: ["low", "plan_low", "explore_low", "medium", "high"].index(x))
        self.logger.info(f"Hierarchy Level: {highest_hierarchy_level}")
        self.hierarchy_keys = list(self.hierarchy.keys())