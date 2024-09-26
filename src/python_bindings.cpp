#include "engine_core.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include "externals/pybind11/include/pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(firesim, m) {
    py::class_<Action, std::shared_ptr<Action>>(m, "Action");

    py::class_<DroneAction, Action, std::shared_ptr<DroneAction>>(m, "DroneAction")
            .def(py::init<>())
            .def(py::init<double, double, int>())
            .def("GetSpeedX", &DroneAction::GetSpeedX)
            .def("GetSpeedY", &DroneAction::GetSpeedY)
            .def("GetWaterDispense", &DroneAction::GetWaterDispense);

    py::class_<State, std::shared_ptr<State>>(m, "State");

    py::class_<DroneState, State, std::shared_ptr<DroneState>>(m, "DroneState")
            .def(py::init<std::pair<double, double>, std::pair<double, double>, std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<int>>, std::vector<std::vector<double>>, std::pair<int, int>, std::pair<int, int>, int, double >())
            .def("SetVelocity", &DroneState::SetVelocity)
            .def("GetVelocity", &DroneState::GetVelocity)
            .def("GetDroneViewNorm", &DroneState::GetDroneViewNorm)
            .def("GetVelocityNorm", &DroneState::GetVelocityNorm)
            .def("GetNewVelocity", &DroneState::GetNewVelocity)
            .def("GetExplorationMap", &DroneState::GetExplorationMap)
            .def("GetFireMap", &DroneState::GetFireMap)
            .def("GetExplorationMapNorm", &DroneState::GetExplorationMapNorm)
            .def("GetPositionNorm", &DroneState::GetPositionNorm)
            .def("GetFireStatus", &DroneState::GetFireStatus)
            .def("GetWaterDispense", &DroneState::GetWaterDispense)
            .def_property_readonly("velocity", &DroneState::get_velocity)
            .def_property_readonly("drone_view", &DroneState::get_drone_view)
            .def_property_readonly("exploration_map", &DroneState::get_map)
            .def_property_readonly("fire_map", &DroneState::get_fire_map)
            .def_property_readonly("position", &DroneState::get_position)
            .def_property_readonly("orientation_vector", &DroneState::get_orientation_vector);

    py::class_<EngineCore>(m, "EngineCore")
            .def_static("GetInstance", &EngineCore::GetInstance)
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
            .def("GetRLStatusFromModel", &EngineCore::GetRLStatusFromModel)
            .def("AgentIsRunning", &EngineCore::AgentIsRunning)
            .def("InitialModeSelectionDone", &EngineCore::InitialModeSelectionDone)
            .def("GetViewRange", &EngineCore::GetViewRange)
            .def("GetTimeSteps", &EngineCore::GetTimeSteps)
            .def("Step", &EngineCore::Step);
}