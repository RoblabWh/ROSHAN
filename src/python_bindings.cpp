#include "engine_core.h"
#include "externals/pybind11/include/pybind11/pybind11.h"

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

    py::class_<State, std::shared_ptr<State>> BaseState(m, "State");

    py::class_<AgentState, State, std::shared_ptr<AgentState>>(m, "DroneState")
            .def(py::init<>())
            .def("GetVelocity", &AgentState::GetVelocity)
            .def("GetDroneViewNorm", &AgentState::GetDroneViewNorm)
            .def("GetVelocityNorm", &AgentState::GetVelocityNorm)
            .def("GetExplorationMap", &AgentState::GetExplorationMap)
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
            .def_property_readonly("velocity", &AgentState::get_velocity)
            .def_property_readonly("drone_view", &AgentState::get_drone_view)
            .def_property_readonly("total_drone_view", &AgentState::get_total_drone_view)
            .def_property_readonly("exploration_map", &AgentState::get_map)
            .def_property_readonly("fire_map", &AgentState::get_fire_map)
            .def_property_readonly("position", &AgentState::get_position)
            .def_property_readonly("orientation_vector", &AgentState::get_orientation_vector);

    py::class_<EngineCore>(m, "EngineCore")
            .def(py::init<>())
            .def("Init", &EngineCore::Init, py::arg("mode"), py::arg("map_path") = "")
            .def("Clean", &EngineCore::Clean)
            .def("Render", &EngineCore::Render)
            .def("Update", &EngineCore::Update)
            .def("HandleEvents", &EngineCore::HandleEvents)
            .def("IsRunning", &EngineCore::IsRunning)
            .def("GetObservations", &EngineCore::GetObservations)
            .def("GetUserInput", &EngineCore::GetUserInput)
            .def("SendDataToModel", &EngineCore::SendDataToModel)
            .def("SendRLStatusToModel", &EngineCore::SendRLStatusToModel)
            .def("UpdateReward", &EngineCore::UpdateReward)
            .def("GetRLStatusFromModel", &EngineCore::GetRLStatusFromModel)
            .def("AgentIsRunning", &EngineCore::AgentIsRunning)
            .def("InitialModeSelectionDone", &EngineCore::InitialModeSelectionDone)
            .def("GetViewRange", &EngineCore::GetViewRange)
            .def("GetTimeSteps", &EngineCore::GetTimeSteps)
            .def("GetMapSize", &EngineCore::GetMapSize)
            .def("Step", &EngineCore::Step);
}