#ifndef PLUGIN_STOP_PHYSICS_HH
#define PLUGIN_STOP_PHYSICS_HH

#include <ros/console.h>

// Gazebo
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo {

  class StopPhysicsPlugin : public WorldPlugin {

    public:
      StopPhysicsPlugin();
      ~StopPhysicsPlugin();
      void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf);

    private:
      physics::WorldPtr world;

  };
  GZ_REGISTER_WORLD_PLUGIN(StopPhysicsPlugin)
}
#endif
