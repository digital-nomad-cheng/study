from mininet.topo import SingleSwitchTopo
from mininet.net import Mininet
from mininet.cli import CLI

net = Mininet(SingleSwitchTopo(2))
net.start()
CLI(net)
net.stop()
