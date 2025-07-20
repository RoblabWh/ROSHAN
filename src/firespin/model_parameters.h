//
// Created by nex on 13.06.23.
//

#ifndef ROSHAN_MODEL_PARAMETERS_H
#define ROSHAN_MODEL_PARAMETERS_H

#include <cmath>
#include <random>
#include <utility>
#include <SDL.h>
#include <yaml-cpp/yaml.h>

class FireModelParameters {

public:
    FireModelParameters() = default;

    void init(const std::string& yaml_path) {
        YAML::Node config = YAML::LoadFile("/home/nex/Dokumente/Code/ROSHAN/parameter_config.yaml");
        auto firemodel = config["fire_model"];
        // Rendering
        auto rendering = firemodel["rendering"];
        render_grid_ = rendering["render_grid"].as<bool>();
        has_noise_ = rendering["has_noise"].as<bool>();
        lingering_ = rendering["lingering"].as<bool>();
        noise_level_ = rendering["noise_level"].as<int>();
        noise_size_ = rendering["noise_size"].as<int>();
        background_color_ = {rendering["background_color"][0].as<Uint8>(),
                             rendering["background_color"][1].as<Uint8>(),
                             rendering["background_color"][2].as<Uint8>(),
                             rendering["background_color"][3].as<Uint8>()};
        // Simulation parameters
        // Timing
        auto simulation = firemodel["simulation"];
        auto time = simulation["time"];
        dt_ = time["dt"].as<double>();
        min_dt_ = time["min_dt"].as<double>();
        max_dt_ = time["max_dt"].as<double>();
        use_default_ = config["settings"]["use_default"].as<bool>();

        // Cells
        auto cells = simulation["cells"];
        cell_ignition_threshold_ = cells["cell_ignition_threshold"].as<double>();
        cell_burning_duration_ = cells["cell_burning_duration"].as<double>();
        cell_size_ = cells["cell_size"].as<double>();
        flood_duration_ = cells["flood_duration"].as<double>();

        // Particles
        auto particles = simulation["particles"];
        // Convective particles
        emit_convective_ = particles["emit_convective"].as<bool>();
        virtualparticle_y_st_ = particles["virtualparticle_y_st"].as<double>();
        virtualparticle_y_lim_ = particles["virtualparticle_y_lim"].as<double>();
        virtualparticle_fl_ = particles["virtualparticle_fl"].as<double>();
        virtualparticle_c0_ = particles["virtualparticle_c0"].as<double>();
        virtualparticle_tau_mem_ = particles["virtualparticle_tau_mem"].as<double>();
        Lt_ = particles["Lt"].as<double>();
        // Radiation particles
        emit_radiation_ = particles["emit_radiation"].as<bool>();
        radiationparticle_y_st_ = particles["radiationparticle_y_st"].as<double>();
        radiationparticle_y_lim_ = particles["radiationparticle_y_lim"].as<double>();

        // Grid
        default_map_ = config["paths"]["init_map"].as<std::string>();
        auto grid = simulation["grid"];
        std_grid_nx_ = grid["uniform_nx"].as<int>();
        std_grid_ny_ = grid["uniform_ny"].as<int>();
        grid_nx_ = std_grid_nx_;
        grid_ny_ = std_grid_ny_;
        exploration_map_size_ = grid["exploration_map_size"].as<int>();
        exploration_map_show_size_ = exploration_map_size_;

        // Wind
        auto wind = simulation["wind"];
        wind_uw_ = wind["wind_uw"].as<double>();
        auto degrees_wind_angle = wind["wind_angle"].as<double>();
        if (degrees_wind_angle == -1.0) {
            // If the wind angle is -1, we set it to a random angle
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 360.0);
            degrees_wind_angle = dis(gen);
        }
        wind_angle_ = degrees_wind_angle * M_PI / 180.0; // Convert degrees to radians
        wind_a_ = wind["wind_a"].as<double>();

        // Environment parameters
        auto environment = config["environment"];
        auto fire_behaviour = environment["fire_behaviour"];
        frame_skips_ = environment["frame_skips"].as<int>();
        ignite_single_cells_ = fire_behaviour["ignite_single_cells"].as<bool>();
        fire_percentage_ = fire_behaviour["fire_percentage"].as<float>();
        fire_spread_prob_ = fire_behaviour["fire_spread_prob"].as<float>();
        fire_noise_ = fire_behaviour["fire_noise"].as<float>();
        recharge_time_active_ = fire_behaviour["recharge_time_active"].as<bool>();

        // Agent parameters
        auto agent = environment["agent"];

        auto flyagent = agent["fly_agent"];
        number_of_flyagents_ = flyagent["num_agents"].as<int>();
        fly_agent_speed_ = flyagent["max_speed"].as<double>();
        fly_agent_view_range_ = flyagent["view_range"].as<int>();

        auto exploreagent = agent["explore_agent"];
        number_of_explorers_ = exploreagent["num_agents"].as<int>();
        explore_agent_speed_ = exploreagent["max_speed"].as<double>();
        explore_agent_view_range_ = exploreagent["view_range"].as<int>();

        auto planneragent = agent["planner_agent"];
        number_of_extinguishers_ = planneragent["num_agents"].as<int>();
        extinguisher_speed_ = planneragent["max_speed"].as<double>();
        extinguisher_view_range_ = planneragent["view_range"].as<int>();

        water_capacity_ = agent["water_capacity"].as<int>();
        extinguish_all_fires_ = agent["extinguish_all_fires"].as<bool>();

        auto agent_behaviour = environment["agent_behaviour"];
        groundstation_start_percentage_ = agent_behaviour["groundstation_start_percentage"].as<float>();
        corner_start_percentage_ = agent_behaviour["corner_start_percentage"].as<float>();
        fire_goal_percentage_ = agent_behaviour["fire_goal_percentage"].as<float>();
    }

    //
    //Flags
    //
    bool map_is_uniform_{};
    bool use_default_{};
    bool corine_loaded_ = false;
    bool initial_mode_selection_done_ = false;
    bool episode_termination_indicator_ = true;

    [[nodiscard]] bool InitialModeSelectionDone() const {return initial_mode_selection_done_;}
    void SetCorineLoaded(bool loaded) {corine_loaded_ = loaded;}
    [[nodiscard]] bool GetCorineLoaded() const {return corine_loaded_;}

    //
    //Render parameters
    //

    bool render_grid_{};
    bool has_noise_{};
    bool lingering_{};
    int noise_level_{};
    int noise_size_{};
    SDL_Color background_color_{};

    //
    // Simulation parameters
    //

    // Timing Parameters
    double dt_{}; // in seconds (s)
    [[nodiscard]] double GetDt() const {return dt_;}
    // Minimum and maximum values for the ImGui Sliders for the simulation parameters
    double min_dt_{}; // in seconds (s)
    double max_dt_{}; // in seconds (s)

    // Parameters for the Cells
    double cell_ignition_threshold_{}; // Time that the cell needs to be visited by a particle to be ignited in seconds (s)
    double cell_burning_duration_{}; // Time the cell is capable of burning in seconds (s)
    double cell_size_{}; // We assume quadratic cells and this is the length of both sides of the cell in meters (m)
    double flood_duration_{}; // Flooding duration of a cell in seconds (s)
    [[nodiscard]] double GetIgnitionDelayTime() const {return cell_ignition_threshold_;}
    [[nodiscard]] double GetCellBurningDuration() const {return cell_burning_duration_;}
    [[nodiscard]] double GetCellSize() const {return cell_size_;} // in meters (m)
    [[nodiscard]] double GetFloodDuration() const {return flood_duration_;}

    // Parameters for the virtual particles
    bool emit_convective_{};
    double virtualparticle_y_st_{}; // Hotness of the Particle (no real world unit yet)
    double virtualparticle_y_lim_{};  // How long the Particle is able to cause ignition (no real world unit yet)
    double virtualparticle_fl_{};  // Scaling factor for new position (need to calibrate)
    double virtualparticle_c0_{}; // A constant close to 2
    double virtualparticle_tau_mem_{}; // A few tens of seconds
    double Lt_{}; // Height of emitting source (m)

    [[nodiscard]] double GetYStVirt() const {return virtualparticle_y_st_;}
    [[nodiscard]] double GetYLimVirt() const {return virtualparticle_y_lim_;}
    [[nodiscard]] double GetFlVirt() const {return virtualparticle_fl_;}
    [[nodiscard]] double GetC0Virt() const {return virtualparticle_c0_;}
    [[nodiscard]] double GetTauMemVirt() const {return virtualparticle_tau_mem_;}
    [[nodiscard]] double GetLt() const {return Lt_;}

    // Parameters for the radiation particles
    bool emit_radiation_{};
    double radiationparticle_y_st_{};
    double radiationparticle_y_lim_{};
    [[nodiscard]] double GetYStRad() const {return radiationparticle_y_st_;}
    [[nodiscard]] double GetYLimRad() const {return radiationparticle_y_lim_;}

    // Parameter for the Grid
    std::string default_map_{};
    // Number of cells in the x direction (rows)
    int std_grid_nx_{};
    int grid_nx_{};
    // Number of cells in the y direction (cols)
    int std_grid_ny_{};
    int grid_ny_{};
    void SetGridNxNy(int nx, int ny) {grid_nx_ = nx; grid_ny_ = ny;}
    void SetGridNxNyStd() {grid_nx_ = std_grid_nx_; grid_ny_ = std_grid_ny_;}
    [[nodiscard]] int GetGridNx() const {return grid_nx_;}
    [[nodiscard]] int GetGridNy() const {return grid_ny_;}
    [[nodiscard]] int GetExplorationTime() const {return grid_nx_ * grid_ny_;}

    int exploration_map_size_{};
    int exploration_map_show_size_{};
    [[nodiscard]] int GetExplorationMapSize() const {return exploration_map_size_;}
    [[nodiscard]] int GetFireMapSize() const {return exploration_map_size_;}

    // Parameters for the wind
    double wind_uw_{}; // The 10-m wind speed in m/s
    double wind_angle_{};
    double wind_a_{}; // The component of the wind speed in the 1st direction
    [[nodiscard]] double GetWindSpeed() const {return wind_uw_;}
    [[nodiscard]] double GetAngle() const {return wind_angle_;}
    void SetWindAngle(double angle) {wind_angle_ = angle;}
    [[nodiscard]] double GetA() const {return wind_a_;}

    void ConvertRealToGridCoordinates(double x, double y, int &i, int &j) const {
        // round x and y to get the cell coordinates
        i = static_cast<int>(std::floor(x / this->GetCellSize()));
        j = static_cast<int>(std::floor(y / this->GetCellSize()));
    }

    void ConvertGridToRealCoordinates(int x_grid, int y_grid, double &x_real, double &y_real) const {
        x_real = x_grid * this->GetCellSize();
        y_real = y_grid * this->GetCellSize();
    }


    //
    // Environment Controls
    //
    bool ignite_single_cells_{};
    float fire_goal_percentage_{}; // in percent (%)
    float fire_percentage_{}; // in percent (%)
    float fire_spread_prob_{}; // in percent (%)
    float fire_noise_{};
    bool recharge_time_active_{};

    // Parameters for the groundstation
    float groundstation_start_percentage_{};
    float corner_start_percentage_{};

    // Parameters for the agent
    int frame_skips_{};
    std::string hierarchy_type;
    void SetHierarchyType(std::string type) { hierarchy_type = std::move(type);}
    [[nodiscard]] std::string GetHierarchyType() const {return hierarchy_type;}
    bool agent_is_running_ = false;
    void SetAgentIsRunning(bool running) {agent_is_running_ = running;}
    [[nodiscard]] bool GetAgentIsRunning() const {return agent_is_running_;}
    int current_env_steps_ = 0;
    [[nodiscard]] int GetCurrentEnvSteps() const {return current_env_steps_;}
    void SetCurrentEnvSteps(int steps) {current_env_steps_ = steps;}
//    int GetTotalEnvSteps() const {return (int)((grid_nx_ * grid_ny_ * (0.1 / dt_)) + 80);}
    [[nodiscard]] int GetTotalEnvSteps() const {
        int agent_factor = hierarchy_type == "fly_agent" ? 1 : hierarchy_type == "explore_agent" ? 10 : 10;
        auto max_speed = hierarchy_type == "fly_agent" ? fly_agent_speed_ : hierarchy_type == "explore_agent" ? explore_agent_speed_ : extinguisher_speed_;
        return (int)(agent_factor * sqrt(grid_nx_ * grid_nx_ + grid_ny_ * grid_ny_) * (20 / (max_speed * dt_)));
    }

    int number_of_flyagents_{};
    double fly_agent_speed_{};
    int fly_agent_view_range_{};
    int number_of_explorers_{};
    double explore_agent_speed_{};
    int explore_agent_view_range_{};
    int number_of_extinguishers_{};
    double extinguisher_speed_{};
    int extinguisher_view_range_{};
    int water_capacity_{};
    bool extinguish_all_fires_{};
    [[nodiscard]] int GetNumberOfFlyAgents() const {return number_of_flyagents_;}
    [[nodiscard]] int GetNumberOfExplorers() const {return number_of_explorers_;}
    [[nodiscard]] int GetNumberOfExtinguishers() const {return number_of_extinguishers_;}
    void SetNumberOfDrones(int number) { number_of_flyagents_ = number;}
    void SetNumberOfExplorers(int number) {number_of_explorers_ = number;}
    void SetNumberOfExtinguishers(int number) {number_of_extinguishers_ = number;}
    [[nodiscard]] int GetWaterCapacity() const {return water_capacity_;}
    [[nodiscard]] double GetWaterRefillDt() const {return GetWaterCapacity() / (5 * 60 / GetDt());}
};


#endif //ROSHAN_MODEL_PARAMETERS_H
