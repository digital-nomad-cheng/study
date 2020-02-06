#ifndef INTERSECTION_H
#define INTERSECTION_H

#include <vector>
#include <future>
#include <mutex>
#include <memory>
#include "traffic_object.hpp"
#include "traffic_light.hpp"

// forward declarations to avoid include cycle
class Street;
class Vehicle;

// auxiliary class to queue and dequeue waiting vehicles in a thread-safe manner
class WaitingVehicles
{
public:
    // getters / setters
    int getSize();

    // typical behaviour methods
    void pushBack(std::shared_ptr<Vehicle> vehicle, std::promise<void> &&promise);
    void permitEntryToFirstInQueue();

private:
    // list of all vehicles waiting to enter this intersection
    std::vector<std::shared_ptr<Vehicle>> _vehicles;  
    std::vector<std::promise<void>> _promises; // list of associated promises
    std::mutex _mutex;
};

class Intersection : public TrafficObject
{
public:
    // constructor / desctructor
    Intersection();

    // getters / setters
    void setIsBlocked(bool isBlocked);

    // typical behaviour methods
    void addVehicleToQueue(std::shared_ptr<Vehicle> vehicle);
    void addStreet(std::shared_ptr<Street> street);
    // return pointer to current list of all outgoing streets
    std::vector<std::shared_ptr<Street>> queryStreets(std::shared_ptr<Street> 
                                                                    incoming); 
    void simulate();
    void vehicleHasLeft(std::shared_ptr<Vehicle> vehicle);
    bool trafficLightIsGreen();

private:

    // typical behaviour methods
    void processVehicleQueue();

    // list of all streets connected to this intersection
    std::vector<std::shared_ptr<Street>> _streets;
    // list of all vehicles and their associated promises waiting to enter 
    // the intersection  
    WaitingVehicles _waitingVehicles; 
    // flag indicating wheter the intersetion is blocked by a vehicle
    bool _isBlocked; 
    TrafficLight _traffic_light;
};

#endif
