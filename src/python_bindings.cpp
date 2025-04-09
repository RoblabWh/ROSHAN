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

    py::class_<DroneState, State, std::shared_ptr<DroneState>>(m, "DroneState")
            .def(py::init<std::pair<double, double>,
                    std::pair<double, double>,
                    std::vector<std::vector<std::vector<int>>>,
                    std::vector<std::vector<double>>,
                    std::vector<std::vector<int>>,
                    std::vector<std::vector<double>>,
                    std::pair<double, double>,
                    std::pair<double, double>,
                    std::pair<double, double>,
                    int,
                    double >())
            .def("SetVelocity", &DroneState::SetVelocity)
            .def("GetVelocity", &DroneState::GetVelocity)
            .def("GetDroneViewNorm", &DroneState::GetDroneViewNorm)
            .def("GetVelocityNorm", &DroneState::GetVelocityNorm)
            .def("GetNewVelocity", &DroneState::GetNewVelocity)
            .def("GetExplorationMap", &DroneState::GetExplorationMap)
            .def("GetFireMap", &DroneState::GetFireMap)
            .def("GetExplorationMapNorm", &DroneState::GetExplorationMapNorm)
            .def("GetPositionNormAroundCenter", &DroneState::GetPositionNormAroundCenter)
            .def("GetGridPositionDoubleNorm", &DroneState::GetGridPositionDoubleNorm)
            .def("GetDeltaGoal", &DroneState::GetDeltaGoal)
            .def("GetOutsideAreaCounter", &DroneState::CountOutsideArea)
            .def("GetOrientationToGoal", &DroneState::GetOrientationToGoal)
            .def("GetGoalPosition", &DroneState::GetGoalPosition)
            .def("GetGoalPositionNorm", &DroneState::GetGoalPositionNorm)
            .def("GetFireView", &DroneState::GetFireView)
            .def("GetWaterDispense", &DroneState::GetWaterDispense)
            .def("GetTotalDroneView", &DroneState::GetTotalDroneView)
            .def_property_readonly("velocity", &DroneState::get_velocity)
            .def_property_readonly("drone_view", &DroneState::get_drone_view)
            .def_property_readonly("total_drone_view", &DroneState::get_total_drone_view)
            .def_property_readonly("exploration_map", &DroneState::get_map)
            .def_property_readonly("fire_map", &DroneState::get_fire_map)
            .def_property_readonly("position", &DroneState::get_position)
            .def_property_readonly("orientation_vector", &DroneState::get_orientation_vector);

    py::class_<EngineCore>(m, "EngineCore")
            .def(py::init<>())
            .def("Init", &EngineCore::Init, py::arg("mode"), py::arg("map_path") = "")
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