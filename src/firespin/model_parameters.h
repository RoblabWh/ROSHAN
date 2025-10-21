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
#include "utils.h"

class FireModelParameters {

public:
    FireModelParameters() = default;

    void init(const std::string& yaml_path) {
        YAML::Node config = YAML::LoadFile(yaml_path);

        // Settings
        llm_support_ = config["settings"]["llm_support"].as<bool>();
        seed_ = config["settings"]["seed"].as<int>();
        if (seed_ == -1) {
            seed_ = static_cast<int>(std::random_device{}());
        }
        gen_.seed(seed_);
        init_rl_mode_ = config["settings"]["rl_mode"].as<std::string>();
        cia_mode_ = config["settings"]["cia_mode"].as<bool>();

        // Paths
        auto paths = config["paths"];
        corine_dataset_name_ = paths["corine_dataset_name"].as<std::string>();

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
        skip_gui_init_ = config["settings"]["skip_gui_init"].as<bool>();

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
            degrees_wind_angle = dis(gen_);
        }
        wind_angle_ = degrees_wind_angle * M_PI / 180.0; // Convert degrees to radians
        wind_a_ = wind["wind_a"].as<double>();

        // Environment parameters
        auto environment = config["environment"];
        auto fire_behaviour = environment["fire_behaviour"];
        num_fire_clusters_ = fire_behaviour["num_fire_clusters"].as<int>();
        fire_percentage_ = fire_behaviour["fire_percentage"].as<float>();
        fire_spread_prob_ = fire_behaviour["fire_spread_prob"].as<float>();
        fire_noise_ = fire_behaviour["fire_noise"].as<float>();
        recharge_time_active_ = fire_behaviour["recharge_time_active"].as<bool>();

        // Agent parameters
        auto agent = environment["agent"];
        drone_size_ = agent["drone_size"].as<double>();
        auto flyagent = agent["fly_agent"];
        number_of_flyagents_ = flyagent["num_agents"].as<int>();
        fly_agent_frame_skips_ = flyagent["frame_skips"].as<int>();
        fly_agent_speed_ = flyagent["max_speed"].as<double>();
        fly_agent_view_range_ = flyagent["view_range"].as<int>();
        fly_agent_time_steps_ = flyagent["time_steps"].as<int>();
        use_simple_policy_ = flyagent["use_simple_policy"].as<bool>();

        auto exploreagent = agent["explore_agent"];
        number_of_explorers_ = exploreagent["num_agents"].as<int>();
        explore_agent_frame_skips_ = exploreagent["frame_skips"].as<int>();
        explore_agent_speed_ = exploreagent["max_speed"].as<double>();
        explore_agent_view_range_ = exploreagent["view_range"].as<int>();
        explore_agent_time_steps_ = exploreagent["time_steps"].as<int>();

        auto planneragent = agent["planner_agent"];
        number_of_extinguishers_ = planneragent["num_agents"].as<int>();
        planner_agent_frame_skips_ = planneragent["frame_skips"].as<int>();
        extinguisher_speed_ = planneragent["max_speed"].as<double>();
        extinguisher_view_range_ = planneragent["view_range"].as<int>();
        planner_agent_time_steps_ = planneragent["time_steps"].as<int>();

        water_capacity_ = agent["water_capacity"].as<int>();

        auto agent_behaviour = environment["agent_behaviour"];
        groundstation_start_percentage_ = agent_behaviour["groundstation_start_percentage"].as<float>();
        std::cout << "Groundstation Start Percentage: " << groundstation_start_percentage_ << std::endl;
        fire_goal_percentage_ = agent_behaviour["fire_goal_percentage"].as<float>();
        std::cout << "Imported all Config Parameters from" << yaml_path << std::endl;
    }

    //Settings
    int seed_{};
    std::string init_rl_mode_{};
    bool cia_mode_{};

    // Paths
    std::string corine_dataset_name_{};

    //
    //Flags
    //
    int mode_{};
    bool map_is_uniform_{};
    bool skip_gui_init_{};
    bool show_small_drones_ = true;
    bool show_drone_circles_ = false;
    bool exit_carefully_ = false;
    bool check_for_model_folder_empty_ = false;
    bool corine_loaded_ = false;
    bool initial_mode_selection_done_ = false;
    bool episode_termination_indicator_ = true;
    bool llm_support_{};
    bool adaptive_start_position_ = false;
    bool adaptive_goal_position_ = false;
    bool random_start_goal_variables_ = false;

    void SetCorineLoaded(bool loaded) {corine_loaded_ = loaded;}
    [[nodiscard]] bool GetCorineLoaded() const {return corine_loaded_;}

    //
    //Render parameters
    //

    bool render_grid_{};
    bool has_noise_{};
    bool lingering_{};
    bool render_particles_=false;
    bool render_terrain_transition=true;
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

    // Random Generator
    std::mt19937 gen_{std::random_device{}()};


    //
    // Environment Controls
    //
    int num_fire_clusters_{};
    float fire_goal_percentage_{}; // in percent (%)
    float fire_percentage_{}; // in percent (%)
    float fire_spread_prob_{}; // in percent (%)
    float fire_noise_{};
    bool recharge_time_active_{};
    bool manual_control_{false};
    int active_drone_{0};

    float SampleFirePercentage() { return std::uniform_real_distribution<float>(0.0, 1.0)(gen_); }
    float SampleFireSpreadProb() { return std::uniform_real_distribution<float>(0.0, 1.0)(gen_); }
    float SampleFireNoise() { return std::uniform_real_distribution<float>(-1.0, 1.0)(gen_); }

    float GetFirePercentage() {return fire_percentage_ < 0 ? this->SampleFirePercentage() : fire_percentage_;}
    float GetFireSpreadProb() {return fire_spread_prob_ < 0 ? this->SampleFireSpreadProb() : fire_spread_prob_;}
    float GetFireNoise() {return fire_noise_ < 0 ? this->SampleFireNoise() : fire_noise_;}

    // Parameters for the groundstation
    float groundstation_start_percentage_{};

    // Parameters for the agent
    int fly_agent_frame_skips_{};
    int explore_agent_frame_skips_{};
    int planner_agent_frame_skips_{};
    std::string hierarchy_type{};
    double drone_size_{};
    void SetHierarchyType(std::string type) { hierarchy_type = std::move(type);}
    [[nodiscard]] std::string GetHierarchyType() const {return hierarchy_type;}
    int current_env_steps_ = 0;
    [[nodiscard]] int GetCurrentEnvSteps() const {return current_env_steps_;}
    void SetCurrentEnvSteps(int steps) {current_env_steps_ = steps;}
//    int GetTotalEnvSteps() const {return (int)((grid_nx_ * grid_ny_ * (0.1 / dt_)) + 80);}
//    [[nodiscard]] int GetTotalEnvSteps() const {
//        int agent_factor = hierarchy_type == "fly_agent" ? 1 : hierarchy_type == "explore_agent" ? 5 : 10;
//        auto max_speed = hierarchy_type == "fly_agent" ? fly_agent_speed_ : hierarchy_type == "explore_agent" ? explore_agent_speed_ : extinguisher_speed_;
//        return (int)(agent_factor * sqrt(grid_nx_ * grid_nx_ + grid_ny_ * grid_ny_) * (20 / (max_speed * dt_)));
//    }

    double coverage_eff = 0.7; // Efficiency factor for exploration agents
    int total_env_steps_ = 0;
    double k_turn_ = 1.5; // Factor to account for turns and non-optimal paths
    double slack_ = 2.0; // Slack factor to allow for exploration and other tasks
    std::string env_step_string_;

    int GetTotalEnvSteps(bool is_eval) {
        const double D = cell_size_ * std::hypot((double)grid_nx_, (double)grid_ny_); // Diagonal distance in meters
        auto max_speed = hierarchy_type == "fly_agent" ? fly_agent_speed_ : hierarchy_type == "explore_agent" ? explore_agent_speed_ : extinguisher_speed_;
        const double base_time = D / max_speed * k_turn_; // Base time in seconds
        int T_physical = (int)std::ceil(base_time / dt_); // Convert to time steps
        int T_Task = T_physical; // default

        if (hierarchy_type == "fly_agent") {
            T_Task = (int)std::ceil(slack_ * T_physical);
            if (use_simple_policy_ && is_eval) {
                const double beta = 0.23;
                const int F = (int)std::ceil(fire_percentage_ * (double)(grid_nx_ * grid_ny_));
                T_Task *= (int)std::ceil(beta * std::sqrt(std::max(1e-9, (double)(grid_nx_ * grid_ny_))) * std::sqrt((double)F) / (max_speed * dt_));
            }

        }
        else if (hierarchy_type == "explore_agent") {
            const double A = (double)grid_nx_ * (double)grid_ny_;
            const double steps_cover = A / std::max(1.0, (double)explore_agent_view_range_ * coverage_eff);
            T_Task = (int)std::ceil(steps_cover);
            T_Task = std::max(T_Task, (int)(slack_ * T_physical));
        }
        else if (hierarchy_type == "planner_agent") {
            const double area_m2 = (grid_nx_ * cell_size_) * (grid_ny_ * cell_size_);
            const double beta = 0.3; // BHH constant
            const double fire_time = 200; // Time a fire burns in seconds
            const int F = (int)std::ceil(fire_percentage_ * (double)(grid_nx_ * grid_ny_));
            double L = beta * std::sqrt(std::max(1e-9, area_m2)) * std::sqrt((double)F);
            double t_move = L / std::max(1e-6, max_speed);
            double t_svc  = F * std::max(0.0, fire_time);
            int steps = (int)std::ceil((t_move + t_svc) / std::max(1e-9, dt_));
            T_Task = std::max(steps, (int)std::ceil(slack_ * T_physical));
        }
        total_env_steps_ = T_Task;
        env_step_string_ = GetTotalEnvStepsExplanation();
        return total_env_steps_;
    }

    [[nodiscard]] std::string GetTotalEnvStepsExplanation() const {
        std::ostringstream oss;
        const double D = cell_size_ * std::hypot((double)grid_nx_, (double)grid_ny_);
        auto max_speed = hierarchy_type == "fly_agent" ? fly_agent_speed_
                                                       : hierarchy_type == "explore_agent" ? explore_agent_speed_
                                                                                           : extinguisher_speed_;
        const double base_time = D / max_speed * k_turn_;
        int T_physical = (int)std::ceil(base_time / dt_);

        oss << "Grid: " << grid_nx_ << " x " << grid_ny_ << " cells ("
            << cell_size_ << " m each)\n";
        oss << "Diagonal distance D = " << D << " m\n";
        oss << "Max speed = " << max_speed << " m/s\n";
        oss << "Turn factor k_turn = " << k_turn_ << "\n";
        oss << "Base time = D / speed * k_turn = " << base_time << " s\n";
        oss << "Physical steps = ceil(base_time / dt) = " << T_physical << "\n";
        oss << "Slack factor = " << slack_ << "\n";

        if (hierarchy_type == "fly_agent") {
            oss << "\nAgent Type: Fly\n";
            oss << "Total steps = ceil(slack * T_physical) = "
                << (int)std::ceil(slack_ * T_physical);
        }
        else if (hierarchy_type == "explore_agent") {
            oss << "\nAgent Type: Explore\n";
            const double A = (double)grid_nx_ * (double)grid_ny_;
            const double steps_cover = A / std::max(1.0, (double)explore_agent_view_range_ * coverage_eff);
            oss << "Area A = " << A << " cells\n";
            oss << "Coverage steps = A / (view_range * efficiency) = "
                << steps_cover << "\n";
            int T_Task = (int)std::ceil(steps_cover);
            T_Task = std::max(T_Task, (int)(slack_ * T_physical));
            oss << "Total steps = max(coverage, slack*physical) = " << T_Task;
        }
        else if (hierarchy_type == "planner_agent") {
            oss << "\nAgent Type: Planner\n";
            const double area_m2 = (grid_nx_ * cell_size_) * (grid_ny_ * cell_size_);
            const double beta = 0.72;
            const double fire_time = 300;
            const int F = (int)std::ceil(fire_percentage_ * (double)(grid_nx_ * grid_ny_));
            double L = beta * std::sqrt(std::max(1e-9, area_m2)) * std::sqrt((double)F);
            double t_move = L / std::max(1e-6, max_speed);
            double t_svc  = F * std::max(0.0, fire_time);
            int steps = (int)std::ceil((t_move + t_svc) / std::max(1e-9, dt_));
            int T_Task = std::max(steps, (int)std::ceil(slack_ * T_physical));
            oss << "Area = " << area_m2 << " m²\n";
            oss << "Fires F = " << F << "\n";
            oss << "TSP estimate length = " << L << " m\n";
            oss << "Move time = " << t_move << " s, Service time = " << t_svc << " s\n";
            oss << "Steps = ceil((move+service)/dt) = " << steps << "\n";
            oss << "Total steps = max(steps, slack*physical) = " << T_Task;
        }
        else {
            oss << "\nAgent Type: Unknown\n";
        }

        return oss.str();
    }

    int number_of_flyagents_{};
    double fly_agent_speed_{};
    int fly_agent_view_range_{};
    int fly_agent_time_steps_{};
    bool use_simple_policy_{};
    int number_of_explorers_{};
    double explore_agent_speed_{};
    int explore_agent_view_range_{};
    int explore_agent_time_steps_{};
    int number_of_extinguishers_{};
    double extinguisher_speed_{};
    int extinguisher_view_range_{};
    int planner_agent_time_steps_{};
    int water_capacity_{};
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
