#include "utils.h"

std::string CellStateToString(CellState cell_state) {
    switch (cell_state) {
        case GENERIC_UNBURNED:
            return "GENERIC_UNBURNED";
        case SEALED:
            return "SEALED";
        case WOODY_NEEDLE_LEAVED_TREES:
            return "WOODY_NEEDLE_LEAVED_TREES";
        case WOODY_BROADLEAVED_DECIDUOUS_TREES:
            return "WOODY_BROADLEAVED_DECIDUOUS_TREES";
        case WOODY_BROADLEAVED_EVERGREEN_TREES:
            return "WOODY_BROADLEAVED_EVERGREEN_TREES";
        case LOW_GROWING_WOODY_PLANTS:
            return "LOW_GROWING_WOODY_PLANTS";
        case PERMANENT_HERBACEOUS:
            return "PERMANENT_HERBACEOUS";
        case PERIODICALLY_HERBACEOUS:
            return "PERIODICALLY_HERBACEOUS";
        case LICHENS_AND_MOSSES:
            return "LICHENS_AND_MOSSES";
        case NON_AND_SPARSLEY_VEGETATED:
            return "NON_AND_SPARSLEY_VEGETATED";
        case WATER:
            return "WATER";
        case SNOW_AND_ICE:
            return "SNOW_AND_ICE";
        case OUTSIDE_AREA:
            return "OUTSIDE_AREA";
        case GENERIC_BURNING:
            return "GENERIC_BURNING";
        case GENERIC_BURNED:
            return "GENERIC_BURNED";
        case GENERIC_FLOODED:
            return "GENERIC_FLOODED";
        default:
            return "UNKNOWN";
    }
}

std::string formatTime(int total_seconds) {
    const int seconds_per_minute = 60;
    const int minutes_per_hour = 60;
    const int hours_per_day = 24;

    int days = total_seconds / (hours_per_day * minutes_per_hour * seconds_per_minute);
    int hours = (total_seconds / (minutes_per_hour * seconds_per_minute)) % hours_per_day;
    int minutes = (total_seconds / seconds_per_minute) % minutes_per_hour;
//    int seconds = total_seconds % seconds_per_minute;

    std::string formatted_time = "";

    if (days > 0) {
        formatted_time += std::to_string(days) + " day(s) ";
    }
    if (hours > 0 || days > 0) {
        formatted_time += std::to_string(hours) + " hour(s) ";
    }
    if (minutes > 0 || hours > 0 || days > 0) {
        formatted_time += std::to_string(minutes) + " minute(s) ";
    }
    //formatted_time += std::to_string(seconds) + " second(s)";

    return formatted_time;
}
