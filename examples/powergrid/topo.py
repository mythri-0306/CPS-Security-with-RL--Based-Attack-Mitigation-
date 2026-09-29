"""
Power Grid Frequency Control CPS Testbed - topo.py

Defines the Mininet network topology:
Single OpenFlow/OVS switch 's1' connected to:
- sensor host (192.168.1.10)
- controller host (192.168.1.20)
- attacker host (192.168.1.77)
"""

from mininet.topo import Topo
from utils import IP, MAC, NETMASK


class PowerGridTopo(Topo):
    """Star topology with sensor, controller, and attacker hosts on switch s1."""

    def build(self):
        # Add single central switch
        switch = self.addSwitch('s1')

        # Add Sensor host
        sensor = self.addHost(
            'sensor',
            ip=IP['sensor'] + NETMASK,
            mac=MAC['sensor']
        )
        self.addLink(sensor, switch)

        # Add Controller host
        controller = self.addHost(
            'controller',
            ip=IP['controller'] + NETMASK,
            mac=MAC['controller']
        )
        self.addLink(controller, switch)

        # Add Attacker host (dormant in Phase 1, present for reachability)
        attacker = self.addHost(
            'attacker',
            ip=IP['attacker'] + NETMASK,
            mac=MAC['attacker']
        )
        self.addLink(attacker, switch)


if __name__ == "__main__":
    from mininet.net import Mininet
    from mininet.node import OVSBridge
    from mininet.cli import CLI

    print("INFO: Building and testing PowerGridTopo in standalone mode...")
    topo = PowerGridTopo()
    net = Mininet(topo=topo, switch=OVSBridge, controller=None)
    net.start()
    print("Testing network reachability:")
    net.pingAll()
    CLI(net)
    net.stop()
