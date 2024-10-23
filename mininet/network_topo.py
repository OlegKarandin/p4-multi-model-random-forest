from p4utils.mininetlib.network_API import NetworkAPI

net = NetworkAPI()

# Network general options
net.setLogLevel('info')
net.setCompiler(p4rt=True) #Useful when we implement Tables
net.execScript('python control_plane.py', reboot=True) #Reboot=True means that when we reboot the topology, this script
#also needs to be rebooted
net.enableCli()

# Network definition
net.addP4RuntimeSwitch('s1')
net.setP4Source('s1','../p4/p4_code_RF_models.p4')
#net.setP4SourceAll() --> To set same .p4 to all switches
net.addHost('h1')
net.addHost('h2')
net.addLink('s1', 'h1')
net.addLink('s1', 'h2')

# Assignment strategy
net.l2()

# Nodes general options
net.enablePcapDumpAll()
net.enableLogAll()

# Start network
net.startNetwork()