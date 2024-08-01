//
// Created by nex on 13.06.23.
//

#ifndef ROSHAN_MODEL_PARAMETERS_H
#define ROSHAN_MODEL_PARAMETERS_H

#include <math.h>
#include <random>
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
    bool map_is_uniform_;


    // Simulation parameters
    double dt_ = 1; // in seconds (s)
    double GetDt() const {return dt_;}
    // Minimum and maximum values for the ImGui Sliders for the simulation parameters
    double min_dt_ = 0.0001; // in seconds (s)
    double max_dt_ = 1.0; // in seconds (s)
    int fire_percentage_ = 1; // in percent (%)


    // Parameters for the virtual particles
    bool emit_convective_ = true;
    double virtualparticle_y_st_ = 1.0; // Hotness of the Particle (no real world unit yet)
    double GetYStVirt() const {return virtualparticle_y_st_;}
    double virtualparticle_y_lim_ = 0.2;  // How long the Particle is able to cause ignition (no real world unit yet)
    double GetYLimVirt() const {return virtualparticle_y_lim_;}
    double virtualparticle_fl_ = 0.15;  // Scaling factor for new position (need to calibrate)
    double GetFlVirt() const {return virtualparticle_fl_;}
    double virtualparticle_c0_ = 1.98; // A constant close to 2
    double GetC0Virt() const {return virtualparticle_c0_;}
    double virtualparticle_tau_mem_ = 10.0; // A few tens of seconds
    double GetTauMemVirt() const {return virtualparticle_tau_mem_;}
    double Lt_ = 80.0; // Height of emitting source (m)
    double GetLt() const {return Lt_;}
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
    double GetYStRad() const {return radiationparticle_y_st_;}
    double radiationparticle_y_lim_ = 0.155;
    double GetYLimRad() const {return radiationparticle_y_lim_;}

    // Parameter for the Grid
    // Number of cells in the x direction (rows)
    int grid_nx_ = 110;
    int GetGridNx() const {return grid_nx_;}
    // Number of cells in the y direction (cols)
    int grid_ny_ = 110;
    int GetGridNy() const {return grid_ny_;}


    // Parameters for the Cells

    // Time that the cell needs to be visited by a particle to be ignited
    double cell_ignition_threshold_ = 100; // in seconds (s)
    double GetIgnitionDelayTime() const {return cell_ignition_threshold_;}
    // Time the cell is capable of burning
    double cell_burning_duration_ = 120; // in seconds (s)
    double GetCellBurningDuration() const {return cell_burning_duration_;}
    // We assume quadratic cells and this is the length of the side of the cell
    double cell_size_ = 10.0; // in meters (m)
    double GetCellSize() const {return cell_size_;} // in meters (m)
    double flood_duration_ = 5.0; // in seconds (s)
    double GetFloodDuration() {return flood_duration_;}
    // Minimum and maximum values for the ImGui Sliders for the cells
    double min_burning_duration_ = 1.0;
    double max_burning_duration_ = 200.0;
    double min_ignition_threshold_ = 1.0;
    double max_ignition_threshold_ = 500.0;
    double min_cell_size_ = 1.0;
    double max_cell_size_ = 100.0;


    // Parameters for the wind
    double wind_uw_ = 10.0; // The 10-m wind speed in m/s
    double GetWindSpeed() const {return wind_uw_;}
    // random number between 0 and 2pi
    double wind_angle_ = random() * 2 * M_PI / RAND_MAX;
    double GetAngle() const {return wind_angle_;}
    double wind_a_ = 0.4; // The component of the wind speed in the 1st direction
    double GetA() const {return wind_a_;}
    // Minimum and maximum values for the ImGui Sliders for the wind
    const double min_Uw_ = 0.0;
    const double max_Uw_ = 35.0; // Hurricane
    const double min_A_ = 0.2;
    const double max_A_ = 0.5;

    void ConvertRealToGridCoordinates(double x, double y, int &i, int &j) const {
        // round x and y to get the cell coordinates
        i = int(trunc(x / this->GetCellSize()));
        j = int(trunc(y / this->GetCellSize()));
    }

    void ConvertGridToRealCoordinates(int x_grid, int y_grid, double &x_real, double &y_real) const {
        x_real = x_grid * this->GetCellSize();
        y_real = y_grid * this->GetCellSize();
    }

    // Deprecated (USE WITH CAUTON!)
    void ConvertRealToGridCoordinatesDrone(double x, double y, int &i, int &j) {
        // round x and y to get the cell coordinates
        i = int(round(x / GetCellSize()));
        j = int(round(y / GetCellSize()));
    }

    // Parameters for the agent
    int number_of_drones_ = 1;
    int view_range_ = 8;
    int GetViewRange() const {return view_range_;}
    int time_steps_ = 16;
    int GetTimeSteps() const {return time_steps_;}
    // std::pair<double, double> min_velocity_ = std::make_pair(-5.0, -5.0);
    // std::pair<double, double> GetMinVelocity() const {return min_velocity_;}
    std::pair<double, double> max_velocity_ = std::make_pair(5.0, 5.0); // X and Y Speed
    // std::pair<double, double> max_velocity_ = std::make_pair(5.0, 2 * M_PI); // Max Speed and Angle
    std::pair<double, double> GetMaxVelocity() const {return max_velocity_;}
    int GetNumberOfDrones() const {return number_of_drones_;}
    void SetNumberOfDrones(int number) {number_of_drones_ = number;}
    double GetDroneSpeed(double speed) { return speed * GetDt(); }

    //Parameters for ImGui
    int RewardsBufferSize = 300;
    int GetRewardsBufferSize() const {return RewardsBufferSize;}

};


#endif //ROSHAN_MODEL_PARAMETERS_H
