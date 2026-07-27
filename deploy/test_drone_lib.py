import intercept_common as ic
from mocap import MocapReceiver, MocapTfPublisher
from drone import CrazyflieDrone, DronePosePublisher
import cflib.crtp
import logging
import time
import warnings

# Configure logging before importing modules that create loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)


# Hide the known cflib warning when firmware still uses legacy hover packet type.
warnings.filterwarnings(
    'ignore',
    message=r'Using legacy TYPE_HOVER_LEGACY\. Please update your crazyflie-firmware\.',
    category=DeprecationWarning,
    module=r'cflib\.crazyflie\.commander',
)

warnings.filterwarnings(
    "ignore",
    message="The supervisor subsystem requires CRTP protocol version 12"
)

if __name__ == "__main__":
    pursuer_config = ic.load_drone_config_from_yaml('config_v2.yaml', 'pursuer')
    evader_config = ic.load_drone_config_from_yaml('config_v2.yaml', 'evader')
    mocap_config = ic.load_mocap_config_from_yaml('config_v2.yaml')

    cflib.crtp.init_drivers()

    drone_pose_publisher = DronePosePublisher(world_frame="world")

    pursuer = CrazyflieDrone(pursuer_config, pose_publisher=drone_pose_publisher)
    evader = CrazyflieDrone(evader_config, pose_publisher=drone_pose_publisher)

    mocap_tf_publisher = MocapTfPublisher(world_frame="world")
    mocap_receiver = MocapReceiver(mocap_config, tf_publisher=mocap_tf_publisher)

    try:
        pursuer.connect()
        pursuer.setup()
        evader.connect()
        evader.setup()

        mocap_receiver.register(rigid_body_id=31, drone=pursuer)
        mocap_receiver.register(rigid_body_id=32, drone=evader)
        # mocap_receiver.register_marker(marker_id=50002, drone=drone)
        mocap_receiver.start()

        pursuer.arm()
        evader.arm()
        pursuer.takeoff(0.75)
        evader.takeoff(0.75)

        pursuer_position_hold = (1.0, 0.0, 0.75, 0.0)
        evader_position_hold = (1.0, 0.0, 0.75, 0.0)

        while pursuer.connected and evader.connected:
            try:
                # state = pursuer.get_state()
                # print(f"Drone state: {state}")

                pursuer.publish_pose()
                evader.publish_pose()

                # dummy CTBR
                # ctbr_command = ic.CTBRCommand(
                #     body_rate_deg=ic.torch.Tensor([0.0, 0.0, 0.0]),
                #     thrust_ratio=ic.torch.Tensor([0.5]),
                #     thrust_pwm=ic.torch.Tensor([50000])
                # )

                # drone.send_ctbr(ctbr_command)

                pursuer.send_position_setpoint(*pursuer_position_hold)
                evader.send_position_setpoint(*evader_position_hold)
                # drone.send_hover_setpoint(position_hold[2])
                time.sleep(pursuer.control_dt)
            except KeyboardInterrupt:
                break
    finally:
        if pursuer.armed:
            pursuer.land()
            pursuer.disarm()
        if evader.armed:
            evader.land()
            evader.disarm()

        mocap_tf_publisher.close()
        mocap_receiver.stop()
        pursuer.disconnect()
        evader.disconnect()
