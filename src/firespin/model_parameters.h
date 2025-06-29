//
// Created by nex on 13.06.23.
//

#ifndef ROSHAN_MODEL_PARAMETERS_H
#define ROSHAN_MODEL_PARAMETERS_H

#include <cmath>
#include <random>
#include <utility>
#include <SDL.h>
class FireModelParameters {

public:
    FireModelParameters() = default;

    //Render parameters
    bool render_grid_ = false;
    bool has_noise_ = true;
    bool lingering_ = true;
    int noise_level_ = 20;
    int noise_size_ = 2;
    SDL_Color background_color_ = {41, 49, 51, 255};
    bool map_is_uniform_{};

    bool corine_loaded_ = false;
    bool initial_mode_selection_done_ = false;
    bool episode_termination_indicator_ = true;
    [[nodiscard]] bool InitialModeSelectionDone() const {return initial_mode_selection_done_;}
    void SetCorineLoaded(bool loaded) {corine_loaded_ = loaded;}
    [[nodiscard]] bool GetCorineLoaded() const {return corine_loaded_;}

    // Simulation parameters
    double dt_ = 0.3; // in seconds (s)
    [[nodiscard]] double GetDt() const {return dt_;}
    // Minimum and maximum values for the ImGui Sliders for the simulation parameters
    double min_dt_ = 0.0001; // in seconds (s)
    double max_dt_ = 5.0; // in seconds (s)

    // Environment Controls
    bool ignite_single_cells_ = false;
    float fire_goal_percentage_ = 0.0 ; // in percent (%)
    float fire_percentage_ = 0.5; // in percent (%)
    float fire_spread_prob_ = 0.095; // in percent (%)
    float fire_noise_ = 1;
    bool recharge_time_active_ = false;


    // Parameters for the virtual particles
    bool emit_convective_ = true;
    double virtualparticle_y_st_ = 1.0; // Hotness of the Particle (no real world unit yet)
    [[nodiscard]] double GetYStVirt() const {return virtualparticle_y_st_;}
    double virtualparticle_y_lim_ = 0.2;  // How long the Particle is able to cause ignition (no real world unit yet)
    [[nodiscard]] double GetYLimVirt() const {return virtualparticle_y_lim_;}
    double virtualparticle_fl_ = 0.15;  // Scaling factor for new position (need to calibrate)
    [[nodiscard]] double GetFlVirt() const {return virtualparticle_fl_;}
    double virtualparticle_c0_ = 1.98; // A constant close to 2
    [[nodiscard]] double GetC0Virt() const {return virtualparticle_c0_;}
    double virtualparticle_tau_mem_ = 10.0; // A few tens of seconds
    [[nodiscard]] double GetTauMemVirt() const {return virtualparticle_tau_mem_;}
    double Lt_ = 80.0; // Height of emitting source (m)
    [[nodiscard]] double GetLt() const {return Lt_;}
    // Minimum and maximum values for the ImGui Sliders for the virtual particles
    double tau_min = 0.1;
    double tau_max = 100.0;
    double min_Y_st_ = 0.0;
    double max_Y_st_ = 1.0;
    double min_Y_lim_ = 0.1;
    double max_Y_lim_ = 0.3;
    double min_Fl_ = 0.0;
    double max_Fl_ = 10.0;
    double min_C0_ = 1.5;
    double max_C0_ = 2.0;
    double min_Lt_ = 10.0;
    double max_Lt_ = 100.0;


    // Parameters for the radiation particles
    bool emit_radiation_ = true;
    double radiationparticle_y_st_ = 1.0;
    [[nodiscard]] double GetYStRad() const {return radiationparticle_y_st_;}
    double radiationparticle_y_lim_ = 0.165;
    [[nodiscard]] double GetYLimRad() const {return radiationparticle_y_lim_;}

    // Parameter for the Grid
    // Number of cells in the x direction (rows)
    const int std_grid_nx_ = 80;
    int grid_nx_ = std_grid_nx_;
    [[nodiscard]] int GetGridNx() const {return grid_nx_;}
    // Number of cells in the y direction (cols)
    const int std_grid_ny_ = 80;
    int grid_ny_ = std_grid_ny_;
    void SetGridNxNy(int nx, int ny) {grid_nx_ = nx; grid_ny_ = ny;}
    void SetGridNxNyStd() {grid_nx_ = std_grid_nx_; grid_ny_ = std_grid_ny_;}
    [[nodiscard]] int GetGridNy() const {return grid_ny_;}
    [[nodiscard]] int GetExplorationTime() const {return grid_nx_ * grid_ny_;}
    int exploration_map_size_ = 50;
    int exploration_map_show_size_ = exploration_map_size_;
    [[nodiscard]] int GetExplorationMapSize() const {return exploration_map_size_;}
    [[nodiscard]] int GetFireMapSize() const {return exploration_map_size_;}


    // Parameters for the Cells

    // Time that the cell needs to be visited by a particle to be ignited
    double cell_ignition_threshold_ = 100; // in seconds (s)
    [[nodiscard]] double GetIgnitionDelayTime() const {return cell_ignition_threshold_;}
    // Time the cell is capable of burning
    double cell_burning_duration_ = 120; // in seconds (s)
    [[nodiscard]] double GetCellBurningDuration() const {return cell_burning_duration_;}
    // We assume quadratic cells and this is the length of the side of the cell
    double cell_size_ = 10.0; // in meters (m)
    [[nodiscard]] double GetCellSize() const {return cell_size_;} // in meters (m)
    double flood_duration_ = 5.0; // in seconds (s)
    [[nodiscard]] double GetFloodDuration() const {return flood_duration_;}
    // Minimum and maximum values for the ImGui Sliders for the cells
    double min_burning_duration_ = 1.0;
    double max_burning_duration_ = 200.0;
    double min_ignition_threshold_ = 1.0;
    double max_ignition_threshold_ = 500.0;
    double min_cell_size_ = 1.0;
    double max_cell_size_ = 100.0;


    // Parameters for the wind

    double wind_uw_ = 10.0; // The 10-m wind speed in m/s
    [[nodiscard]] double GetWindSpeed() const {return wind_uw_;}
    // random number between 0 and 2pi
    double wind_angle_ = random() * 2 * M_PI / RAND_MAX;
    [[nodiscard]] double GetAngle() const {return wind_angle_;}
    void SetWindAngle(double angle) {wind_angle_ = angle;}
    double wind_a_ = 0.314; // The component of the wind speed in the 1st direction
    [[nodiscard]] double GetA() const {return wind_a_;}
    // Minimum and maximum values for the ImGui Sliders for the wind
    const double min_Uw_ = 0.0;
    const double max_Uw_ = 35.0; // Hurricane
    const double min_A_ = 0.2;
    const double max_A_ = 0.5;

    void ConvertRealToGridCoordinates(double x, double y, int &i, int &j) const {
        // round x and y to get the cell coordinates
//        i = int(trunc(x / this->GetCellSize()));
//        j = int(trunc(y / this->GetCellSize()));
        i = static_cast<int>(std::floor(x / this->GetCellSize()));
        j = static_cast<int>(std::floor(y / this->GetCellSize()));
    }

    void ConvertGridToRealCoordinates(int x_grid, int y_grid, double &x_real, double &y_real) const {
        x_real = x_grid * this->GetCellSize();
        y_real = y_grid * this->GetCellSize();
    }

    // Deprecated (USE WITH CAUTON!)
    [[maybe_unused]] void ConvertRealToGridCoordinatesDrone(double x, double y, int &i, int &j) const {
        // round x and y to get the cell coordinates
        i = int(round(x / GetCellSize()));
        j = int(round(y / GetCellSize()));
    }

    // Parameters for the agent
    std::string hierarchy_type;
    void SetHierarchyType(std::string type) { hierarchy_type = std::move(type);}
    [[nodiscard]] std::string GetHierarchyType() const {return hierarchy_type;}
    bool agent_is_running_ = false;
    void SetAgentIsRunning(bool running) {agent_is_running_ = running;}
    [[nodiscard]] bool GetAgentIsRunning() const {return agent_is_running_;}
//    int total_env_steps_ = 200;
    int current_env_steps_ = 0;
    [[nodiscard]] int GetCurrentEnvSteps() const {return current_env_steps_;}
    void SetCurrentEnvSteps(int steps) {current_env_steps_ = steps;}
//    int GetTotalEnvSteps() const {return (int)((grid_nx_ * grid_ny_ * (0.1 / dt_)) + 80);}
    [[nodiscard]] int GetTotalEnvSteps() const {
        int agent_factor = hierarchy_type == "FlyAgent" ? 1 : hierarchy_type == "ExploreAgent" ? 10 : 10;
        return (int)(agent_factor * sqrt(grid_nx_ * grid_nx_ + grid_ny_ * grid_ny_) * (20 / (max_velocity_.first * dt_)));
    }

    [[nodiscard]] static int GetViewRange(const std::string& agent_type) {
        int view_range_ = agent_type == "FlyAgent" ? 12 : agent_type == "ExploreAgent" ? 12 : 2;
        return view_range_;
    }
    int time_steps_ = 3; // 16 //32
    [[nodiscard]] int GetTimeSteps() const {return time_steps_;}
    // std::pair<double, double> min_velocity_ = std::make_pair(-5.0, -5.0);
    // std::pair<double, double> GetMinVelocity() const {return min_velocity_;}
    std::pair<double, double> max_velocity_ = std::make_pair(10, 10); // X and Y Speed
    // std::pair<double, double> max_velocity_ = std::make_pair(5.0, 2 * M_PI); // Max Speed and Angle
    [[nodiscard]] std::pair<double, double> GetMaxVelocity() const {return max_velocity_;}
    int number_of_drones_ = 1;
    int number_of_explorers = 2;
    int number_of_extinguishers = 4;
    [[nodiscard]] int GetNumberOfExplorers() const {return number_of_explorers;}
    [[nodiscard]] int GetNumberOfDrones() const {return number_of_drones_;}
    [[nodiscard]] int GetNumberOfExtinguishers() const {return number_of_extinguishers;}
    void SetNumberOfDrones(int number) {number_of_drones_ = number;}
    void SetNumberOfExplorers(int number) {number_of_explorers = number;}
    void SetNumberOfExtinguishers(int number) {number_of_extinguishers = number;}
    int water_capacity_ = 10;
    [[nodiscard]] int GetWaterCapacity() const {return water_capacity_;}
    [[nodiscard]] double GetWaterRefillDt() const {return GetWaterCapacity() / (5 * 60 / GetDt());}
    bool extinguish_all_fires_ = false;

    // Parameters for the groundstation
    float groundstation_start_percentage_ = 0.0;
    float corner_start_percentage_ = 0.9;

    //Parameters for ImGui
    int RewardsBufferSize = 300;
    [[nodiscard]] int GetRewardsBufferSize() const {return RewardsBufferSize;}

};


#endif //ROSHAN_MODEL_PARAMETERS_H
