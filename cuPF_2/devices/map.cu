#include "ros/ros.h"
#include "nav_msgs/OccupancyGrid.h"

float *hmap;
int width;
int height;
float resolution;
float2 center;

bool isFirstmapReceived = false;

void mapCallback(const nav_msgs::OccupancyGrid& msg)
{
    /*
    # This represents a 2-D grid map, in which each cell represents the probability of
    # occupancy.

    Header header 
    MapMetaData info
        time map_load_time                  # The time at which the map was loaded
        float32 resolution                  # The map resolution [m/cell]
        uint32 width                        # Map width [cells]
        uint32 height                       # Map height [cells]
        geometry_msgs/Pose origin           # The origin of the map [m, m, rad].  This is the real-world pose of the
                                            # cell (0,0) in the map.
    int8[] data                             # The map data, in row-major order, starting with (0,0).  Occupancy
                                            # probabilities are in the range [0,100].  Unknown is -1.
    */
    width = (int)msg.info.width;
    height = (int)msg.info.height;
    resolution = (float)msg.info.resolution;
    hmap = (float*)malloc(sizeof(float) * width * height);
    center.x = msg.info.origin.position.x;
    center.y = msg.info.origin.position.y;
    for(int i = 0; i < width * height; i++)
    {
        hmap[i] = (float)(msg.data[i]) / 100.0;
    }
    isFirstmapReceived = true;
}