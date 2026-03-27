#include "engine_core.h"
#include "reinforcementlearning/feature_schema.h"
#include "reinforcementlearning/feature_definitions.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include "externals/pybind11/include/pybind11/stl.h"
#include "externals/pybind11/include/pybind11/complex.h"

namespace py = pybind11;

PYBIND11_MODULE(firesim, m) {
    py::class_<Action, std::shared_ptr<Action>> BaseAction(m, "Action");

    py::class_<FlyAction, Action, std::shared_ptr<FlyAction>>(m, "DroneAction")
            .def(py::init<>())
            .def(py::init<double, double>());

    py::class_<ExploreAction, Action, std::shared_ptr<ExploreAction>>(m, "ExploreAction")
            .def(py::init<>())
            .def(py::init<double, double>());

    py::class_<PlanAction, Action, std::shared_ptr<PlanAction>>(m, "PlanAction")
            .def(py::init<>())
            .def(py::init<std::vector<std::pair<double, double>>>());

    py::class_<State, std::shared_ptr<State>> BaseState(m, "State");

    py::class_<AgentState, State, std::shared_ptr<AgentState>>(m, "AgentState")
            .def(py::init<>())
            .def("GetExplorationMapNorm", &AgentState::GetExplorationMapNorm)
            .def("GetGridPositionDoubleNorm", &AgentState::GetGridPositionDoubleNorm)
            .def_property_readonly("velocity", &AgentState::get_velocity)
            .def_property_readonly("drone_view", &AgentState::get_drone_view)
            .def_property_readonly("exploration_map", &AgentState::get_map)
            .def_property_readonly("fire_map", &AgentState::get_fire_map)
            .def_property_readonly("position", &AgentState::get_position)
            .def_property_readonly("water_dispense", &AgentState::get_water_dispense)
            .def_property_readonly("drone_positions", &AgentState::get_drone_positions)
            .def_property_readonly("fire_positions", &AgentState::get_fire_positions)
            .def_property_readonly("goal_positions", &AgentState::get_goal_positions);


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
            .def("GetBatchedObservations", &EngineCore::GetBatchedObservations)
            .def("GetUserInput", &EngineCore::GetUserInput)
            .def("SendDataToModel", &EngineCore::SendDataToModel)
            .def("SendRLStatusToModel", &EngineCore::SendRLStatusToModel)
            .def("UpdateReward", &EngineCore::UpdateReward)
            .def("GetRLStatusFromModel", &EngineCore::GetRLStatusFromModel)
            .def("AgentIsRunning", &EngineCore::AgentIsRunning)
            .def("InitialModeSelectionDone", &EngineCore::InitialModeSelectionDone)
            .def("InitializeMap", &EngineCore::InitializeMap)
            .def("Step", &EngineCore::Step, py::arg("agent_type"), py::arg("actions"), py::arg("skip_observations") = false);

    // Module-level function: returns schema metadata without needing an engine instance
    m.def("GetFeatureSchemaInfo", [](const std::string& agent_type) -> py::dict {
        FeatureSchema schema;
        if (agent_type == "fly_agent" || agent_type == "PlannerFlyAgent" || agent_type == "ExploreFlyAgent")
            schema = CreateFlyAgentSchema();
        else if (agent_type == "planner_agent")
            schema = CreatePlannerAgentSchema();
        else
            throw std::runtime_error("Unknown agent type for schema: " + agent_type);

        py::dict result;
        for (const auto& group : schema.groups) {
            py::dict group_info;
            group_info["dims"] = (group.type == FeatureGroupType::FIXED) ? group.TotalDims() : group.bulk_dims;
            group_info["type"] = (group.type == FeatureGroupType::FIXED) ? "fixed"
                               : (group.type == FeatureGroupType::RELATIONAL) ? "relational" : "set";
            if (group.type == FeatureGroupType::FIXED) {
                py::list columns;
                for (const auto& info : group.GetColumnInfo())
                    columns.append(py::make_tuple(info.first, info.second));
                group_info["columns"] = columns;
            }
            result[py::str(group.name)] = group_info;
        }
        return result;
    }, py::arg("agent_type"), "Return feature schema metadata for the given agent type.");
}