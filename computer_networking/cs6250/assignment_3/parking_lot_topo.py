from mininet.topo import Topo

class ParkingLotTopo(Topo):
    "Parking Lot Topology"

    def __init__(self, n=1, cpu=.1, bw=10, delay=None,
                 max_queue_size=None, **params):
        """Parking lot topology with one receiver
           and n clients.
           n: number of clients
           cpu: system fraction for each host
           bw: link bandwidth in Mb/s
           delay: link delay (e.g. 10ms)"""

        # Initialize topo
        Topo.__init__(self, **params)

        # Host and link configuration
        hconfig = {'cpu': cpu}
        lconfig = {'bw': bw, 'delay': delay,
                   'max_queue_size': max_queue_size }

        # Create the actual topology
        receiver = self.addHost('receiver')

        # Switch ports 1:uplink 2:hostlink 3:downlink
        uplink, hostlink, downlink = 1, 2, 3

        # The following template code creates a parking lot topology
        # TODO: Replace the template code to create a parking lot topology for any arbitrary N (>= 1)
        if n < 1: # network must have at least 1 host
            return -1

        s = [] # Python list of switches
        h = [] # Python list of hosts

        # dynamically add all hosts and switches to network backbone first
        for i in range(n):
            switch_name = 's%s' % (i+1)
            host_name   = 'h%s' % (i+1)

            s.append( self.addSwitch(switch_name) )  # s[0] is switch1
            h.append( self.addHost(host_name) )      # h[0] is host1

            # Wire up clients
            self.addLink(h[i], s[i], port1=0, port2=hostlink, **lconfig)
            
            # link to previous switch
            if i > 0:
                self.addLink(s[i-1], s[i], port1=downlink, port2=uplink, **lconfig)

                
        # Wire up receiver to first switch
        self.addLink(receiver, s[0], port1=0, port2=uplink, **lconfig)

        '''
        # for N = 1
        # Begin: Template code
        s1 = self.addSwitch('s1')
        h1 = self.addHost('h1', **hconfig)

        # Wire up receiver
        self.addLink(receiver, s1, port1=0, port2=uplink, **lconfig)

        # Wire up clients
        self.addLink(h1, s1, port1=0, port2=hostlink, **lconfig)

        # Uncomment the next 8 lines to create a N = 3 parking lot topology
        s2 = self.addSwitch('s2')
        h2 = self.addHost('h2', **hconfig)
        self.addLink(s1, s2,
                     port1=downlink, port2=uplink, **lconfig)
        self.addLink(h2, s2,
                     port1=0, port2=hostlink, **lconfig)

        s3 = self.addSwitch('s3')
        h3 = self.addHost('h3', **hconfig)
        self.addLink(s2, s3,
                     port1=downlink, port2=uplink, **lconfig)
        self.addLink(h3, s3,
                     port1=0, port2=hostlink, **lconfig)
        
        # End: Template code
        '''


