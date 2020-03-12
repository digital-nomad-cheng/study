## Introduction
Project code for CS6250 at Georgia Tech, lectures freely available from Udacity. Work done using Mininet to simulate network environments.

## Preparatioon
1. Download VirtualBox:
    https://www.virtualbox.org/wiki/Downloads
2. Download VirtualMachine according to [syllabs](https://www.omscs.gatech.edu/sites/default/files/documents/course_page_docs/syllabi/cs-6250_computer_networks_syllabus.pdf)
3. Mininet [Documentation](https://github.com/mininet/mininet/wiki/Introduction-to-Mininet#creating)

## Assignments
### Assignment_2 Mininet Topology
+ Run topology.sh and review output.
+ Modify mntopo.py to insert an additional switch between the hosts. Helpful to review Mininet documentation on this.
+ Rerun topology.sh, output should be similar.
+ Test latency by running the ping wrapper, ping.py. Should get ~6ms.
+ Increase the latency delay from 1ms to 10ms in mntopo.py.
+ Re-test latency. Should get ~60ms.
+ Increase the bandwidth from 10Mbps to 50Mbps in mntopo.py.
+ Re-run topology.sh and review output.

