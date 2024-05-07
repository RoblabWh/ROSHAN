//
// Created by nex on 13.07.23.
//

#ifndef ROSHAN_DRONE_H
#define ROSHAN_DRONE_H

#include <utility>
#include <SDL.h>
#include <deque>
#include <memory>
#include "agents/drone_agent/rendering/DroneRenderer.h"
#include "src/models/firespin/model_parameters.h"
#include "src/models/firespin/firemodel_gridmap.h"
#include "drone_state.h"

// TODO Remove circular dependency
class GridMap;

class DroneAgent {
public:
    explicit DroneAgent(std::shared_ptr<SDL_Renderer> renderer, std::pair<int, int> point, FireModelParameters &parameters, int id);
    ~DroneAgent() = default;
    std::deque<DroneState> GetStates() { return drone_states_; }
    void Update(double speed_x, double speed_y, std::vector<std::vector<int>> terrain, std::vector<std::vector<int>> fire_status, std::vector<std::vector<int>> updated_map);
    void Move(double speed_x, double speed_y);
    void MoveByAngle(double netout_speed, double netout_angle);
    bool DispenseWater(GridMap &grid_map);
    std::pair<int, int> GetGridPosition();
    std::pair<double, double> GetGridPositionDouble();
    std::pair<double, double> GetRealPosition();
    void IncrementOutOfAreaCounter() { out_of_area_counter_++; }
    void ResetOutOfAreaCounter() { out_of_area_counter_ = 0; }
    int GetOutOfAreaCounter() { return out_of_area_counter_; }
    DroneState GetLastState() { return drone_states_[0]; }
    int DroneSeesFire();
    int GetId() const { return id_; }
    int GetViewRange() const { return view_range_; }
    void Render(std::pair<int, int> position, int size);
    void Initialize(std::vector<std::vector<int>> terrain, std::vector<std::vector<int>> fire_status, std::pair<int, int> size, double cell_size);
    double FindNearestFireDistance();
private:
    std::pair<double, double> GetRealPositionFromGrid(int x, int y);
    void UpdateStates(double speed_x, double speed_y, std::vector<std::vector<int>> terrain, std::vector<std::vector<int>> fire_status, std::vector<std::vector<int>> updated_map);
    int id_;
    FireModelParameters &parameters_;
    std::deque<DroneState> drone_states_;
    std::pair<double, double> position_; // x, y in (m)
    int view_range_;
    int out_of_area_counter_;
    std::pair<double, double> velocity_; // angular & linear
    DroneRenderer renderer_;

};


#endif //ROSHAN_DRONE_H
