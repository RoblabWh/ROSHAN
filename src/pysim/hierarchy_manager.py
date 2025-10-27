from agent_handler import AgentHandler
from utils import SimulationBridge
import logging


class HierarchyManager:
    def __init__(self, config, sim_bridge : SimulationBridge):
        self.config = config
        self.sim_bridge = sim_bridge
        self.hierarchy = {}
        self.logger = logging.getLogger("HierarchyManager")
        self.max_low_level_steps = 50
        self.build_hierarchy()

    def _should_reset(self, agent) -> bool:
        return (
            agent.hierarchy_early_stop
            or agent.env_reset
            or self.sim_bridge.get("env_reset")
        )

    def _reset_agent(self, agent):
        agent.hierarchy_steps = 0
        agent.hierarchy_early_stop = False
        self.sim_bridge.set("env_reset", False)

    def restruct_current_obs(self, observations_):
        for agent in self.hierarchy.values():
            agent.restruct_current_obs(observations_)

    def _train_high(self, engine):
        high = self.hierarchy["high"]
        if high.hierarchy_steps % self.max_low_level_steps == 0 or self._should_reset(high):
            # Do a Training Step for HighLevel Agent
            high.train_loop(engine=engine)
            self._reset_agent(high)

        # Step through all low level agents (PlanFlyAgent)
        self.hierarchy["plan_low"].eval_loop(engine=engine, evaluate=False)
        high.hierarchy_steps += 1
        self.sim_bridge.set("env_reset", False)

    def _train_medium(self, engine):
        medium = self.hierarchy["medium"]
        if self._should_reset(medium):
            medium.step_without_network(engine=engine)
            self._reset_agent(medium)

        medium.hierarchy_early_stop = all(self.hierarchy["explore_low"].eval_loop(engine=engine, evaluate=False))
        medium.hierarchy_steps += 1

    def _train_low(self, engine):
        low = self.hierarchy["low"]
        low.train_loop(engine=engine)

    def train(self, engine):
        if "high" in self.hierarchy:
            self._train_high(engine)
        elif "medium" in self.hierarchy:
            # Train Loop for MediumLevel Agent TODO Currently not implemented, will only do Exploration and NO Training
            self._train_medium(engine)
        else:
            self._train_low(engine)

    def _eval_high(self, engine):
        medium = self.hierarchy["medium"]
        if medium and self._should_reset(medium):
            medium.step_without_network(engine)

        high = self.hierarchy["high"]
        if high.hierarchy_steps % self.max_low_level_steps == 0 or self._should_reset(high):
            # Do an Evaluation Step for HighLevel Agent
            high.eval_loop(engine=engine, evaluate=True)
            self._reset_agent(high)

        if medium:
            medium.hierarchy_early_stop = all(self.hierarchy["explore_low"].eval_loop(engine=engine, evaluate=False))

        self.hierarchy["plan_low"].eval_loop(engine=engine, evaluate=False)
        high.hierarchy_steps += 1
        self.sim_bridge.set("env_reset", False)

    def _eval_medium(self, engine):
        medium = self.hierarchy["medium"]
        if self._should_reset(medium):
            medium.step_without_network(engine)
            self._reset_agent(medium)

        medium.hierarchy_early_stop = all(self.hierarchy["explore_low"].eval_loop(engine=engine, evaluate=False))
        medium.hierarchy_steps += 1

    def _eval_low(self, engine):
        low = self.hierarchy["low"]
        low.eval_loop(engine=engine, evaluate=True)

    def eval(self, engine):
        if "high" in self.hierarchy:
            self._eval_high(engine)
        elif "medium" in self.hierarchy:
            self._eval_medium(engine)
        else:
            self._eval_low(engine)

    def get_final_metric(self, metric_name):
        if "high" in self.hierarchy:
            return self.hierarchy["high"].get_final_metric(metric_name)
        elif "medium" in self.hierarchy:
            return self.hierarchy["medium"].get_final_metric(metric_name)
        else:
            return self.hierarchy["low"].get_final_metric(metric_name)

    def update_status(self):
        if "high" in self.hierarchy:
            self.hierarchy["high"].update_status()
        elif "medium" in self.hierarchy:
            self.hierarchy["medium"].update_status()
        else:
            self.hierarchy["low"].update_status()

    def build_hierarchy(self):
        # Create the top level Agent Object
        agent_handler = AgentHandler(
             config=self.config,
             agent_type=self.config["settings"]["hierarchy_type"],
             mode=self.config["settings"]["rl_mode"],
             sim_bridge=self.sim_bridge,
             is_sub_agent=False
        )
        agent_handler.load_model(change_status=True)

        self.hierarchy[agent_handler.hierarchy_level] = agent_handler
        construct_medium = agent_handler.hierarchy_level in {"medium", "high"}

        # Construct a low level agent if the current agent is a high level agent
        if agent_handler.hierarchy_level == "high":
            planner_fly_agent = AgentHandler(
                config=self.config,
                sim_bridge=self.sim_bridge,
                agent_type="fly_agent",
                subtype="PlannerFlyAgent",
                mode="eval"
            )
            planner_fly_agent.hierarchy_level = "plan_low"
            planner_fly_agent.load_model()
            self.hierarchy["plan_low"] = planner_fly_agent

        if construct_medium:
            # Construct a medium level agent if the current agent is a high level agent
            if agent_handler.hierarchy_level == "high":
                medium_level_agent = AgentHandler(
                      config=self.config,
                      sim_bridge=self.sim_bridge,
                      agent_type="explore_agent",
                      mode="eval"
                )
                self.hierarchy["medium"] = medium_level_agent
            explore_fly_agent = AgentHandler(
                   config=self.config,
                   sim_bridge=self.sim_bridge,
                   agent_type="fly_agent",
                   subtype="ExploreFlyAgent",
                   mode="eval"
            )
            explore_fly_agent.load_model()
            self.hierarchy["explore_low"] = explore_fly_agent

        highest_hierarchy_level = max(
            self.hierarchy.keys(),
            key=lambda x: ["low", "plan_low", "explore_low", "medium", "high"].index(x)
        )
        self.logger.info(f"Hierarchy Level: {highest_hierarchy_level}")