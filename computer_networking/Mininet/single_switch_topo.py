from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel

class SingleSwitchTopo(Topo):
    """Single swicth connected to n hosts."""
    def build(self, n=2):
        switch = self.addSwitch('s1')
        for i in range(1, n+1):
            host = self.addHost("h%s" % i)
            self.addLink(host, switch)

def simpleTest():
    """Create and test a simple network"""
    topo = SingleSwitchTopo(n=4)
    net = Mininet(topo)
    net.start()
    print("Dumping host connections")
    dumpNodeConnections(net.hosts)
    print("Testing network connectivity")
    net.pingAll()
    net.stop()

if __name__ == "__main__":
    setLogLevel('info')
    simpleTest()

