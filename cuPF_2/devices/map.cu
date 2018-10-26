#include "ros/ros.h"
#include "nav_msgs/OccupancyGrid.h"

float *h_map;       // ホスト側の地図
float *d_map;		// デバイス側の地図
int width;
int height;
float resolution;
float2 center;
bool isFirstmapReceived = false;

// ホストからデバイスへ地図を転送
static void SetupLMap(float * from, size_t count)
{
	cudaMalloc((void **)&d_map, count);
	cudaMemcpyToSymbol(d_map, from, count);
}

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
    h_map = (float*)malloc(sizeof(float) * width * height);
    center.x = msg.info.origin.position.x;
    center.y = msg.info.origin.position.y;
    for(int i = 0; i < width * height; i++)
    {
        h_map[i] = (float)(msg.data[i]) / 100.0;
    }
    // 尤度マップ転送(host->device)
	SetupLMap(h_map, sizeof(float) * width * height);
    isFirstmapReceived = true;
}