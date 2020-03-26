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

### Assignment_3 Parking Lot
[!assignment_3.jpg]()
+ Complete the` __init__` function of the ParkingLotTopo class to generalize this for any number of hosts, n>0. The resulting topology is as shown in the figure above.
+ Complete the `run_parkinglot_expt` function to generate TCP flows using iperf.
+ Final result is running sudo ./parkinglot-sweep.sh to test various parameters of n=1, 2, 3, 4, 5.
+ Run my additional submit.sh to collect the output of parkinglot-sweep.sh wrapper per submission specifications. Submit all bwm.txt files.
+ Complete additional quiz questions in quiz.txt.

### Assignment_4 Static and Learning Switch

