"""Base classes for backend of ROS environments."""
import time
import rospy
from rospy.timer import Timer
from std_msgs.msg import Bool

class BaseEpisodeMonitor(object):
    """Base class for episode monitors.
    This class set up a ROS node for monitoring an episode of interaction in a
    ROS environment.

    """
    def __init__(self, *args, **kwargs):
        rospy.init_node('EpisodeMonitor')

        # Subscribers
        rospy.Subscriber('/rl/simulator_restart', Bool, self.restart_callback)

        # Publishers
        self.is_running = False
        # This is an event-driven state signal
        self.is_running_pub = rospy.Publisher(
            "/rl/is_running", Bool, queue_size=10, latch=True)
        # This is periodic `episode_done` signal
        self.heartbeat_pub = rospy.Publisher(
            "/rl/simulator_heartbeat", Bool, queue_size=10, latch=True)
        Timer(rospy.Duration(1/20.0), self.__heartbeat)

    def terminate(self):
        """Terminate the episode.
        This method terminates the episode and set proper signals to talk to the
        ros_environment. It first set the `is_running` flag to False so that
        ros_environment knows the episode is done. Then the `_terminate` method
        is called to do the heavy liftings. And finally a `False` signal is
        published to signal env_node shutdown for ROS environment.

        The child classes should implement `self._terminate()` for this method
        to work as intended.

        :return:
        """
        # Set simulator state to False before termination.
        self.is_running = False

        # flush heartbeat for 3 seconds so that the signal won't be missed
        secs = 3
        while secs != 0:
            print "[BaseEpisodeMonitor.terminate()]: Shutdown simulator nodes in {} secs".format(secs)
            secs -= 1
            time.sleep(1.0)

        # Work, work, work!
        self._terminate()

        # signal env node shutdown after termination.
        print "[BaseEpisodeMonitor.terminate()]: publish heartbeat=False!"
        self.is_running_pub.publish(False)

    def restart_callback(self, data):
        """Terminate and restart simulator on call. If the data received is
        `False` then only terminates the episode without a restart; Otherwise
        start a new episode.

        :param data: data received on topic '/rl/simulator_restart'.
        :return:
        """

        print ("[BaseEpisodeMonitor.restart]: "
               "restart callback with {}").format(data.data)

        # terminates simulator if data is False
        if not data.data:
            print "[BaseEpisodeMonitor.restart]: mere termination requested."
            self.terminate()
            print "[BaseEpisodeMonitor.restart]: termination finished."
            return
        # restart if data is True
        else:
            self._start()
            print("[BaseEpisodeMonitor.restart]: restarted launch file!")
            self.is_running = True
            print "[BaseEpisodeMonitor.restart]: publishing heartbeat=True!"
            self.is_running_pub.publish(True)
            print "[BaseEpisodeMonitor.restart]: signaling env node up!"

    def spin(self):
        rospy.spin()
        self.terminate()

    def _terminate(self):
        """This guy does all the real work."""
        raise NotImplementedError(
            '[BaseEpisodeMonitor]: _terminate() not implemented.'
        )

    def _start(self):
        raise NotImplementedError(
            '[BaseEpisodeMonitor]: _start() not implemented.'
        )

    def __heartbeat(self, *args, **kwargs):
        self.heartbeat_pub.publish(self.is_running)