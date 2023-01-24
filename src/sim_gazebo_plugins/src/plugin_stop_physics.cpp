#include "plugin_stop_physics.h"

namespace gazebo
{

    // Constructor
    StopPhysicsPlugin::StopPhysicsPlugin() : WorldPlugin()
    {
        ROS_INFO("StopPhysicsPlugin has been instantiated.");
    }

    // Destructor
    StopPhysicsPlugin::~StopPhysicsPlugin()
    {
    }

    void StopPhysicsPlugin::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
        ROS_INFO("StopPhysicsPlugin load callback function.");
        if (_world->PhysicsEnabled())
        {
            ROS_INFO("StopPhysicsPlugin is turning off physics.");
            _world->SetPhysicsEnabled(false);
        }
    }

} // end of namespace gazebo
