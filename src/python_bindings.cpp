#include "engine_core.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include "externals/pybind11/include/pybind11/stl.h"
#include "externals/pybind11/include/pybind11/complex.h"

namespace py = pybind11;

PYBIND11_MODULE(firesim, m) {
    py::class_<Action, std::shared_ptr<Action>> BaseAction(m, "Action");

    py::class_<FlyAction, Action, std::shared_ptr<FlyAction>>(m, "DroneAction")
            .def(py::init<>())
            .def(py::init<double, double>())
            .def("GetSpeedX", &FlyAction::GetSpeedX)
            .def("GetSpeedY", &FlyAction::GetSpeedY)
            .def("GetWaterDispense", &FlyAction::GetWaterDispense);

    py::class_<ExploreAction, Action, std::shared_ptr<ExploreAction>>(m, "ExploreAction")
            .def(py::init<>())
            .def(py::init<double, double>())
            .def("GetGoalX", &ExploreAction::GetGoalX)
            .def("GetGoalY", &ExploreAction::GetGoalY);

    py::class_<PlanAction, Action, std::shared_ptr<PlanAction>>(m, "PlanAction")
            .def(py::init<>())
            .def(py::init<std::vector<std::pair<double, double>>>())
            .def("GetGoalFromAction", &PlanAction::GetGoalFromAction);

    py::class_<State, std::shared_ptr<State>> BaseState(m, "State");

    py::class_<AgentState, State, std::shared_ptr<AgentState>>(m, "AgentState")
            .def(py::init<>())
            .def("GetVelocity", &AgentState::GetVelocity)
            .def("GetDroneViewNorm", &AgentState::GetDroneViewNorm)
            .def("GetVelocityNorm", &AgentState::GetVelocityNorm)
            .def("GetExplorationMap", &AgentState::GetExplorationMap)
            .def("GetExploredFires", &AgentState::GetExploredFires)
            .def("GetFireMap", &AgentState::GetFireMap)
            .def("GetMultipleTotalDroneView", &AgentState::GetMultipleTotalDroneView)
            .def("GetExplorationMapNorm", &AgentState::GetExplorationMapNorm)
            .def("GetExplorationMapScalar", &AgentState::GetExplorationMapScalar)
            .def("GetPositionNormAroundCenter", &AgentState::GetPositionNormAroundCenter)
            .def("GetGridPositionDoubleNorm", &AgentState::GetGridPositionDoubleNorm)
            .def("GetDeltaGoal", &AgentState::GetDeltaGoal)
            .def("GetOutsideAreaCounter", &AgentState::CountOutsideArea)
            .def("GetOrientationToGoal", &AgentState::GetOrientationToGoal)
            .def("GetGoalPosition", &AgentState::GetGoalPosition)
            .def("GetGoalPositionNorm", &AgentState::GetGoalPositionNorm)
            .def("GetFireView", &AgentState::GetFireView)
            .def("GetWaterDispense", &AgentState::GetWaterDispense)
            .def("GetTotalDroneView", &AgentState::GetTotalDroneView)
            .def("GetDronePositions", &AgentState::GetDronePositions)
            .def("GetFirePositions", &AgentState::GetFirePositions)
            .def("GetGoalPositions", &AgentState::GetGoalPositions)
            .def("GetDistancesToOtherAgents", &AgentState::GetDistancesToOtherAgents)
            .def("GetDistancesMask", &AgentState::GetDistancesMask)
            .def("GetCosSinToGoal", &AgentState::GetCosSinToGoal)
            .def("GetSpeed", &AgentState::GetSpeed)
            .def("GetDistanceToGoal", &AgentState::GetDistanceToGoal)
            .def("GetID", &AgentState::GetID)
            .def_property_readonly("velocity", &AgentState::get_velocity)
            .def_property_readonly("drone_view", &AgentState::get_drone_view)
            .def_property_readonly("total_drone_view", &AgentState::get_total_drone_view)
            .def_property_readonly("exploration_map", &AgentState::get_map)
            .def_property_readonly("explored_fires", &AgentState::get_explored_fires)
            .def_property_readonly("fire_map", &AgentState::get_fire_map)
            .def_property_readonly("position", &AgentState::get_position)
            .def_property_readonly("orientation_vector", &AgentState::get_orientation_vector)
            .def_property_readonly("goal_position", &AgentState::get_goal_position)
            .def_property_readonly("map_dimensions", &AgentState::get_map_dimensions)
            .def_property_readonly("cell_size", &AgentState::get_cell_size)
            .def_property_readonly("perfect_goals", &AgentState::get_perfect_goals)
            .def_property_readonly("water_dispense", &AgentState::get_water_dispense)
            .def_property_readonly("multiple_total_drone_views", &AgentState::get_multiple_total_drone_view)
            .def_property_readonly("drone_positions", &AgentState::get_drone_positions)
            .def_property_readonly("fire_positions", &AgentState::get_fire_positions)
            .def_property_readonly("goal_positions", &AgentState::get_goal_positions)
            .def_property_readonly("distances_to_other_agents", &AgentState::get_distances_to_other_agents)
            .def_property_readonly("distances_to_other_agents_mask", &AgentState::get_distances_to_other_agents_mask)
            .def_property_readonly("get_id", &AgentState::get_id);


    py::enum_<TerminationKind>(m, "TerminationKind")
            .value("None", TerminationKind::None)
            .value("Failed", TerminationKind::Failed)
            .value("Succeeded", TerminationKind::Succeeded);

    py::enum_<FailureReason>(m, "FailureReason")
            .value("None", FailureReason::None)
            .value("Timeout", FailureReason::Timeout)
            .value("BoundaryExit", FailureReason::BoundaryExit)
            .value("Burnout", FailureReason::Burnout)
            .value("Collision", FailureReason::Collision)
            .value("Stuck", FailureReason::Stuck)
            .value("NoProgress", FailureReason::NoProgress)
            .value("Other", FailureReason::Other);

    py::class_<AgentTerminal>(m, "AgentTerminal")
            .def_readonly("is_terminal", &AgentTerminal::is_terminal)
            .def_readonly("kind",        &AgentTerminal::kind)
            .def_readonly("reason",      &AgentTerminal::reason);

    py::class_<EpisodeSummary>(m, "EpisodeSummary")
            .def_readonly("env_reset",       &EpisodeSummary::env_reset)
            .def_readonly("any_failed",      &EpisodeSummary::any_failed)
            .def_readonly("any_succeeded",   &EpisodeSummary::any_succeeded)
            .def_readonly("explorers_reached_goal", &EpisodeSummary::explorers_reached_goal)
            .def_readonly("reason",          &EpisodeSummary::reason);

    py::class_<StepResult>(m, "StepResult")
            .def_readonly("observations",   &StepResult::observations)
            .def_readonly("rewards",        &StepResult::rewards)
            .def_readonly("terminals",      &StepResult::terminals)
            .def_readonly("summary",        &StepResult::summary)
            .def_readonly("percent_burned", &StepResult::percent_burned);

    py::class_<EngineCore>(m, "EngineCore")
            .def(py::init<>())
            .def("Init", &EngineCore::Init, py::arg("mode"), py::arg("config_path") = "../config.yaml")
            .def("Clean", &EngineCore::Clean)
            .def("Render", &EngineCore::Render)
            .def("Update", &EngineCore::Update)
            .def("HandleEvents", &EngineCore::HandleEvents)
            .def("IsRunning", &EngineCore::IsRunning)
            .def("GetObservations", &EngineCore::GetObservations)
            .def("GetBatchedFlyObservations", &EngineCore::GetBatchedFlyObservations)
            .def("GetBatchedPlannerObservations", &EngineCore::GetBatchedPlannerObservations)
            .def("GetUserInput", &EngineCore::GetUserInput)
            .def("SendDataToModel", &EngineCore::SendDataToModel)
            .def("SendRLStatusToModel", &EngineCore::SendRLStatusToModel)
            .def("UpdateReward", &EngineCore::UpdateReward)
            .def("GetRLStatusFromModel", &EngineCore::GetRLStatusFromModel)
            .def("AgentIsRunning", &EngineCore::AgentIsRunning)
            .def("InitialModeSelectionDone", &EngineCore::InitialModeSelectionDone)
            .def("InitializeMap", &EngineCore::InitializeMap)
            .def("Step", &EngineCore::Step, py::arg("agent_type"), py::arg("actions"), py::arg("skip_observations") = false);
}